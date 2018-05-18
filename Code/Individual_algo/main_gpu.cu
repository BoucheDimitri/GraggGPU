#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**************************************************************
The code in time.h is a part of a course on cuda taught by its authors:
Lokman A. Abbas-Turki
**************************************************************/
#include "timer.h"


// Compare function for qsort
int compare_function(const void *a,const void *b) {
    float *x = (float *) a;
		float *y = (float *) b;
    if (*x < *y) return - 1;
    else if (*x > *y) return 1;
    return 0;
}


// Generate gaussian vector using Box Muller
void gaussian_vector(float *v, float mu, float sigma, int n) {

    for (int i = 0; i<n; i++){
		    float u1 = (float)rand()/(float)(RAND_MAX);
		    float u2 = (float)rand()/(float)(RAND_MAX);
		    v[i] = sigma * (sqrtf( -2 * logf(u1)) * cosf(2 * M_PI * u2)) + mu;
	  }
}


//Function to print a small vector of floats on host
void print_vector(float *c, int m, int n) {

    for (int i=0; i<m; i++){
        printf("%f     ", c[i]);
        printf("\n");
 	  }
}


// Kernel for computing the square of a vector (INPLACE)
// We actually only need z ** 2 in the computations and not z
// The square norm is also computed
__global__ void square_kernel(float *zsqrGPU, float *znorm, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while(idx < n){
        float zi = zsqrGPU[idx];
        float zsqr_i = zi * zi;
		    zsqrGPU[idx] = zi * zi;
        atomicAdd(znorm, zsqr_i);
		    idx += gridDim.x * blockDim.x;
	  }
}


// Device function for computing f (the secular function of interest) at a given point x
__device__ float secfunc(float *dGPU, float *zsqrGPU, float rho, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        sum += zsqrGPU[i] / (dGPU[i] - x);
	  }

    return rho + sum;
}


// Device function for computing f' (the prime derivative of the secular function of interest) at a given point x
__device__ float secfunc_prime(float *dGPU, float *zsqrGPU, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        int di = dGPU[i];
		    sum += zsqrGPU[i] / ((di - x) * (di - x));
    }

	  return sum;
}


// Device function for computing f'' (the second derivative of the secular function of interest)
__device__ float secfunc_second(float *dGPU, float *zsqrGPU, float x, int n){
    float sum = 0;

		for (int i = 0; i < n; i++) {
		    float di = dGPU[i];
				sum += zsqrGPU[i] / ((di - x) * (di - x) * (di - x));
		}

		return 2 * sum;
}


// Useful intermediary function, see equations (30) and (31) from Li's paper on page 13 and equation (42) on page 20
__device__ float discrimant_int(float a, float b, float c){

    if (a <= 0) return (a - sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a + sqrtf(a * a - 4 * b *c));
}


// Useful intermediary function, see equation (46) from Li's paper on page 21
__device__ float discrimant_ext(float a, float b, float c){

    if (a >= 0) return (a + sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a - sqrtf(a * a - 4 * b *c));
}


// h partition of the secular function, used for Initialization
__device__ float h_secfunc(float d_k, float d_kplus1, float zsqr_k, float zsqr_kplus1, float x){

    return zsqr_k / (d_k - x) + zsqr_kplus1 / (d_kplus1 - x);
}


// Initialization for interior roots (see section 4 of Li's paper - initial guesses from page 18)
__device__ float initialization_int(float *dGPU, float *zsqrGPU, float rho, int k, int n){

    float d_k = dGPU[k];
    float d_kplus1 = dGPU[k + 1];
    float zsqr_k = zsqrGPU[k];
    float zsqr_kplus1 = zsqrGPU[k + 1];
    float middle = (d_k + d_kplus1) / 2;
    float delta = d_kplus1 - d_k;
    float f = secfunc(dGPU, zsqrGPU, rho, middle, n);
    float c = f - h_secfunc(d_k, d_kplus1, zsqr_k, zsqr_kplus1, middle);

    if (f >= 0){
        float a = c * delta + zsqr_k + zsqr_kplus1;
        float b = zsqr_k * delta;
        return discrimant_int(a, b, c) + d_k;
    }

    else {
        float a = - c * delta + zsqr_k + zsqr_kplus1;
        float b = - zsqr_kplus1 * delta;
        return discrimant_int(a, b, c) + d_kplus1;
    }
}


// Initialization for the exterior root (see section 4 of Li's paper - initial guesses from page 18)
__device__ float initialization_ext(float *dGPU, float *zsqrGPU, float *znorm, float rho, int n){

    float d_nminus1 = dGPU[n - 1];
    float d_nminus2 = dGPU[n - 2];
    float d_n = d_nminus1 + znorm[0] / rho;
    float zsqr_nminus1 = zsqrGPU[n - 1];
    float zsqr_nminus2 = zsqrGPU[n - 2];
    float middle = (d_nminus1 + d_n) / 2;
    float f = secfunc(dGPU, zsqrGPU, rho, middle, n);
    if (f <= 0){
        float hd = h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, d_n);
        float c = f - h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        if (c <= - hd) {
            return d_n;
        }

        else {
            float delta = d_nminus1 - d_nminus2;
            float a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
            float b = - zsqr_nminus1 * delta;
            return discrimant_ext(a, b, c) + d_n;
        }
    }

    else {
        float delta = d_nminus1 - d_nminus2;
        float c = f - h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        float a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
        float b = - zsqr_nminus1 * delta;
        return discrimant_ext(a, b, c) + d_n;
    }
}


// Computation of a from the paper (page 13)
__device__ float a_gragg(float f, float fprime, float delta_k, float delta_kplus1){

    return (delta_k + delta_kplus1) * f - delta_k * delta_kplus1 * fprime;

}


// Computation of b from the paper (page 13)
__device__ float b_gragg(float f, float delta_k, float delta_kplus1){

    return delta_k * delta_kplus1 * f;
}


// Computation of c from the section Gragg of the paper (page 15)
__device__ float c_gragg(float f, float fprime, float fsecond, float delta_k, float delta_kplus1){

    return f - (delta_k + delta_kplus1) * fprime + delta_k * delta_kplus1 * fsecond / 2.0;

}


// Compute of the update for x (eta) for the interior roots (see section 3.1 - Iteration fomulas, pages 12 and 13)
__device__ float eta_int(float d_k, float d_kplus1, float f, float fprime, float fsecond, float x, int k, int n){

    float delta_k = d_k - x;
    float delta_kplus1 = d_kplus1 - x;
    float a = a_gragg(f, fprime, delta_k, delta_kplus1);
    float b = b_gragg(f, delta_k, delta_kplus1);
    float c = c_gragg(f, fprime, fsecond, delta_k, delta_kplus1);
    float eta = discrimant_int(a, b, c);
    return eta;
}

// Compute of the update of x (+eta) for the exterior root
__device__ float eta_ext(float d_nminus2, float d_nminus1, float f, float fprime, float fsecond, float x, int n){

    float delta_nminus2 = d_nminus2 - x;
    float delta_nminus1 = d_nminus1 - x;
    float a = a_gragg(f, fprime, delta_nminus2, delta_nminus1);
    float b = b_gragg(f, delta_nminus2, delta_nminus1);
    float c = c_gragg(f, fprime, fsecond, delta_nminus2, delta_nminus1);
    float eta = discrimant_ext(a, b, c);
    return eta;
}

// Iterate to find the k-th interior root
__device__ float find_root_int(float *dGPU, float *zsqrGPU, float rho, float x, int k, int n, int maxit, float epsilon){

    int i = 0;
    float f = secfunc(dGPU, zsqrGPU, rho, x, n);;
    float d_k = dGPU[k];
    float d_kplus1 = dGPU[k + 1];

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(dGPU, zsqrGPU, rho, x, n);
        float fprime = secfunc_prime(dGPU, zsqrGPU, x, n);
        float fsecond = secfunc_second(dGPU, zsqrGPU, x, n);
        float eta = eta_int(d_k, d_kplus1, f, fprime, fsecond, x, k, n);
        x += eta;
        i ++;
    }

    if (k == 0){
        printf("\n");
        printf("********************* CONTROLS ********************* \n");
        printf("We print the first, the last and 10 %% of the interior eigenvalues as a check \n");
    }

    if (k%(int)(n/10) == 0){
        printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n", k, x, f, i);
    }

    return x;
}


// Iterate to  find the last root (the exterior one)
__device__ float find_root_ext(float *dGPU, float *zsqrGPU, float rho, float x, int n, int maxit, float epsilon){

    int i = 0;
    float d_nminus2 = dGPU[n - 2];
    float d_nminus1 = dGPU[n - 1];
    float f = secfunc(dGPU, zsqrGPU, rho, x, n);

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(dGPU, zsqrGPU, rho, x, n);
        float fprime = secfunc_prime(dGPU, zsqrGPU, x, n);
        float fsecond = secfunc_second(dGPU, zsqrGPU, x, n);
        float eta = eta_ext(d_nminus2, d_nminus1, f, fprime, fsecond, x, n);
        x += eta;
        i ++;
    }
    // Print the last eigen value
    printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n", n - 1, x, f, i);
    return x;
}


// Kernel to launch and distribute the searching of roots among GPU cores
__global__ void find_roots_kernel(float *xstarGPU, float *x0GPU, float *dGPU, float *zsqrGPU, float *znorm, float rho, int n, int maxit, int epsilon){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

		// First core gets search of the last root (the exterior one)
    if (idx == 0){
        float x = x0GPU[n - 1];
        xstarGPU[n - 1] = find_root_ext(dGPU, zsqrGPU, rho, x, n, maxit, epsilon);
    }

		// Each next core searches one interval (interior interval)
    else {
        while (idx < n) {
            float x = x0GPU[idx - 1];
            xstarGPU[idx - 1] = find_root_int(dGPU, zsqrGPU, rho, x, idx - 1, n, maxit, epsilon);
						// in case we have not launched enough cores to cover all intervals
          	idx += gridDim.x * blockDim.x;
        }
    }
}



// Kernel to compute the initial guesses from the paper on GPU
__global__ void initialize_x0_kernel(float *x0GPU, float *dGPU, float *zsqrGPU, float *znorm, float rho, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

		// First core compute the initial guess for last root (the exterior one)
    if (idx == 0){
        x0GPU[n - 1] = initialization_ext(dGPU, zsqrGPU, znorm, rho,  n);
    }

		// Each next core compute initial guess for one interval (interior interval)
    else {
    		while (idx < n) {
        		x0GPU[idx - 1] = initialization_int(dGPU, zsqrGPU, rho, idx - 1, n);
        	  idx += gridDim.x * blockDim.x;
        }
    }
}





int main (void) {

    /****************** Declaration ******************/
    // Declare vectors or floats
    float *d, *z, *xstar;


    // rho parameter
    float rho = 10;


    // Size of arrow matrix chosen by the user
    //int n= 10;
    int n;
    printf("\nWhich n (number of roots for the function) do you want? \n");
    scanf("%d", &n);
    printf("\n \n******************* CHOICE OF N ******************** \n");
    printf("n = %d\n", n);

    //Maximum number of iterations
    int maxit = 1e4;


    //Stopping criterion
    float epsilon = 1e-6;


    // Memory allocation
    d = (float*)malloc(n*sizeof(float));
    z = (float*)malloc(n*sizeof(float));
    xstar = (float*)malloc(n*sizeof(float));

    // Create instance of class Timer
    Timer Tim;


    //Fill the vector d with linear function of n
    for (int i=0; i < n; i++){
        d[i] = 2 * n - i;
    }

    // sort the vector in ascending order
    qsort(d, n, sizeof(float), compare_function);

    //print_vector(d, 10, n);
    // Fill a as a vector of gaussian of mean mu and std sigma
    //float mu_d = 0.5 * n;
    //float sigma_d = 0.05 * n;
    //gaussian_vector(d, mu_d, sigma_d, n);
    // We sort by descending order then
    //qsort(d, n, sizeof(float), compare_function);

    //print_vector(d, 10, n);


    //for (int i=0; i < n; i++){
        //z[i] = n - i;
    //}

    // Gaussian rank 1 perturbation
    float mu_z = 5;
    float sigma_z = 1;
    gaussian_vector(z, mu_z, sigma_z, n);

    // Start timer
    Tim.start();

    /***************** GPU memory alloc *****************/

    // Declare vectors on GPU
    float *dGPU, *zsqrGPU, *znorm, *x0GPU, *xstarGPU;

    // Create memory space for vectors on GPU
    cudaMalloc(&dGPU, n*sizeof(float));
    cudaMalloc(&zsqrGPU, n*sizeof(float));
    cudaMalloc(&znorm, sizeof(float));
    cudaMalloc(&x0GPU, n*sizeof(float));
    // Container for the results
    cudaMalloc(&xstarGPU, n*sizeof(float));


    /***************** Transfer on GPU *****************/


    // Transfers on GPU
    cudaMemcpy(dGPU, d, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(zsqrGPU, z, n*sizeof(float), cudaMemcpyHostToDevice);


    // We first compute the square and squared norm
    square_kernel <<<1024, 512>>> (zsqrGPU, znorm, n);


    // Initialization of x0 on GPU
    initialize_x0_kernel <<<1024, 512>>> (x0GPU, dGPU, zsqrGPU, znorm, rho, n);


    /***************** Root computation ****************/
    // Find roots on GPU
    find_roots_kernel <<<1024, 512>>> (xstarGPU, x0GPU, dGPU, zsqrGPU, znorm, rho, n, maxit, epsilon);


    // Transfer results on CPU to print it
    cudaMemcpy(xstar, xstarGPU, n*sizeof(float), cudaMemcpyDeviceToHost);

    // End timer
    Tim.add();

    // Print the first zeros
    // Number of roots to display
    int m = 10;
    printf("\n********************* RESULTS ********************** \n");
    printf("The first %i resulting roots (eigen values) are : \n", m);
    print_vector(xstar, m, n);


    // Print how long it took
    printf("GPU timer for root finding (CPU-GPU and GPU-CPU transfers included) : %f s\n\n", (float)Tim.getsum());

    //print_vector(x0_vec, 10, n);



    // Free memory on GPU
    cudaFree(dGPU);
    cudaFree(zsqrGPU);
    cudaFree(x0GPU);
    cudaFree(xstarGPU);

    // Free memory on CPU
    free(d);
    free(z);
    free(xstar);
}
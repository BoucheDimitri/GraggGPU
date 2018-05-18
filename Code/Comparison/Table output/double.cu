#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**************************************************************
The code in time.h is a part of a course on cuda taught by its authors:
Lokman A. Abbas-Turki
**************************************************************/
#include "timer.h"

/**************************************************************
Common functions
**************************************************************/

// Compare function for qsort
int compare_function(const void *a,const void *b) {
    double *x = (double *) a;
		double *y = (double *) b;
    if (*x < *y) return - 1;
    else if (*x > *y) return 1;
    return 0;
}


// Generate gaussian vector using Box Muller
void gaussian_vector(double *v, double mu, double sigma, int n) {

    for (int i = 0; i<n; i++){
		    double u1 = (double)rand()/(double)(RAND_MAX);
		    double u2 = (double)rand()/(double)(RAND_MAX);
		    v[i] = sigma * (sqrt( -2 * log(u1)) * cos(2 * M_PI * u2)) + mu;
	  }
}


//Function to print a small vector of doubles on host
void print_vector(double *c, int m, int n) {

    for (int i=0; i<m; i++){
        printf("%f     ", c[i]);
        printf("\n");
 	  }
}


// Kernel for computing the square of a vector (INPLACE)
// We actually only need z ** 2 in the computations and not z
// The square norm is also computed
void square_vector(double *z, double *znorm, int n){
		for (int i = 0; i < n; i++) {
				double zi = z[i];
				double zsqri = zi * zi;
				z[i] = zsqri;
				znorm[0] += zsqri;
		}
}


// Function for computing f (the secular function of interest) at a given point x
double secfunc(double *d, double *zsqr, double rho, double x, int n) {

    double sum = 0;
    for (int i=0; i < n; i++){
        sum += zsqr[i] / (d[i] - x);
	  }

    return rho + sum;
}


// Function for computing f' (the prime derivative of the secular function of interest) at a given point x
double secfunc_prime(double *d, double *zsqr, double x, int n) {

    double sum = 0;
    for (int i=0; i < n; i++){
        int di = d[i];
		    sum += zsqr[i] / ((di - x) * (di - x));
    }

	  return sum;
}


// Device function for computing f'' (the second derivative of the secular function of interest)
double secfunc_second(double *d, double *zsqr, double x, int n){
    double sum = 0;

		for (int i = 0; i < n; i++) {
		    double di = d[i];
				sum += zsqr[i] / ((di - x) * (di - x) * (di - x));
		}

		return 2 * sum;
}


// Useful intermediary function, see equations (30) and (31) from Li's paper on page 13 and equation (42) on page 20
double discrimant_int(double a, double b, double c){

    if (a <= 0) return (a - sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a + sqrtf(a * a - 4 * b *c));
}


// Useful intermediary function, see equation (46) from Li's paper on page 21
double discrimant_ext(double a, double b, double c){

    if (a >= 0) return (a + sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a - sqrtf(a * a - 4 * b *c));
}


// h partition of the secular function, used for Initialization
double h_secfunc(double d_k, double d_kplus1, double zsqr_k, double zsqr_kplus1, double x){

    return zsqr_k / (d_k - x) + zsqr_kplus1 / (d_kplus1 - x);
}


// Initialization for interior roots (see section 4 of Li's paper - initial guesses from page 18)
double initialization_int(double *d, double *zsqr, double rho, int k, int n){

    double d_k = d[k];
    double d_kplus1 = d[k + 1];
    double zsqr_k = zsqr[k];
    double zsqr_kplus1 = zsqr[k + 1];
    double middle = (d_k + d_kplus1) / 2;
    double delta = d_kplus1 - d_k;
    double f = secfunc(d, zsqr, rho, middle, n);
    double c = f - h_secfunc(d_k, d_kplus1, zsqr_k, zsqr_kplus1, middle);

    if (f >= 0){
        double a = c * delta + zsqr_k + zsqr_kplus1;
        double b = zsqr_k * delta;
        return discrimant_int(a, b, c) + d_k;
    }

    else {
        double a = - c * delta + zsqr_k + zsqr_kplus1;
        double b = - zsqr_kplus1 * delta;
        return discrimant_int(a, b, c) + d_kplus1;
    }
}


// Initialization for the exterior root (see section 4 of Li's paper - initial guesses from page 18)
double initialization_ext(double *d, double *zsqr, double *znorm, double rho, int n){

    double d_nminus1 = d[n - 1];
    double d_nminus2 = d[n - 2];
    double d_n = d_nminus1 + znorm[0] / rho;
    double zsqr_nminus1 = zsqr[n - 1];
    double zsqr_nminus2 = zsqr[n - 2];
    double middle = (d_nminus1 + d_n) / 2;
    double f = secfunc(d, zsqr, rho, middle, n);
    if (f <= 0){
        double hd = h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, d_n);
        double c = f - h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        if (c <= - hd) {
            return d_n;
        }

        else {
            double delta = d_nminus1 - d_nminus2;
            double a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
            double b = - zsqr_nminus1 * delta;
            return discrimant_ext(a, b, c) + d_n;
        }
    }

    else {
        double delta = d_nminus1 - d_nminus2;
        double c = f - h_secfunc(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        double a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
        double b = - zsqr_nminus1 * delta;
        return discrimant_ext(a, b, c) + d_n;
    }
}


// Computation of a from the paper (page 13)
double a_gragg(double f, double fprime, double delta_k, double delta_kplus1){

    return (delta_k + delta_kplus1) * f - delta_k * delta_kplus1 * fprime;

}


// Computation of b from the paper (page 13)
double b_gragg(double f, double delta_k, double delta_kplus1){

    return delta_k * delta_kplus1 * f;
}


// Computation of c from the section Gragg of the paper (page 15)
double c_gragg(double f, double fprime, double fsecond, double delta_k, double delta_kplus1){

    return f - (delta_k + delta_kplus1) * fprime + delta_k * delta_kplus1 * fsecond / 2.0;

}


// Compute of the update for x (eta) for the interior roots (see section 3.1 - Iteration fomulas, pages 12 and 13)
double eta_int(double d_k, double d_kplus1, double f, double fprime, double fsecond, double x, int k, int n){

    double delta_k = d_k - x;
    double delta_kplus1 = d_kplus1 - x;
    double a = a_gragg(f, fprime, delta_k, delta_kplus1);
    double b = b_gragg(f, delta_k, delta_kplus1);
    double c = c_gragg(f, fprime, fsecond, delta_k, delta_kplus1);
    double eta = discrimant_int(a, b, c);
    return eta;
}

// Compute of the update of x (+eta) for the exterior root
double eta_ext(double d_nminus2, double d_nminus1, double f, double fprime, double fsecond, double x, int n){

    double delta_nminus2 = d_nminus2 - x;
    double delta_nminus1 = d_nminus1 - x;
    double a = a_gragg(f, fprime, delta_nminus2, delta_nminus1);
    double b = b_gragg(f, delta_nminus2, delta_nminus1);
    double c = c_gragg(f, fprime, fsecond, delta_nminus2, delta_nminus1);
    double eta = discrimant_ext(a, b, c);
    return eta;
}

// Iterate to find the k-th interior root
double find_root_int(double *d, double *zsqr, double rho, double x, int k, int n, int maxit, double epsilon, double*loss_CPU){

    int i = 0;
    double f = secfunc(d, zsqr, rho, x, n);;
    double d_k = d[k];
    double d_kplus1 = d[k + 1];

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(d, zsqr, rho, x, n);
        double fprime = secfunc_prime(d, zsqr, x, n);
        double fsecond = secfunc_second(d, zsqr, x, n);
        double eta = eta_int(d_k, d_kplus1, f, fprime, fsecond, x, k, n);
        x += eta;
        i ++;
    }
    *loss_CPU += (double)(abs(f)/n);

    return x;
}


// Iterate to  find the last root (the exterior one)
double find_root_ext(double *d, double *zsqr, double rho, double x, int n, int maxit, double epsilon, double*loss_CPU){

    int i = 0;
    double d_nminus2 = d[n - 2];
    double d_nminus1 = d[n - 1];
    double f = secfunc(d, zsqr, rho, x, n);

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(d, zsqr, rho, x, n);
        double fprime = secfunc_prime(d, zsqr, x, n);
        double fsecond = secfunc_second(d, zsqr, x, n);
        double eta = eta_ext(d_nminus2, d_nminus1, f, fprime, fsecond, x, n);
        x += eta;
        i ++;
    }
    *loss_CPU += (float)(abs(f)/n);
    return x;
}


void find_roots(double *xstar, double *x0, double *d, double *zsqr, double *znorm, double rho, int n, int maxit, int epsilon, double *loss_CPU){
    // We make sure that the loss is set to 0
    *loss_CPU =0;
		for (int i=0; i<n-1; i++){
				xstar[i] = find_root_int(d, zsqr, rho, x0[i], i, n, maxit, epsilon, loss_CPU);
		}

		xstar[n - 1] = find_root_ext(d, zsqr, rho, x0[n - 1], n, maxit, epsilon, loss_CPU);
}


void initialize_x0(double *x0, double *d, double *zsqr, double *znorm, double rho, int n){

		for (int i=0; i<n-1; i++){
				x0[i] = initialization_int(d, zsqr, rho, i, n);
		}

		x0[n - 1] = initialization_ext(d, zsqr, znorm, rho,  n);
}



/**************************************************************
GPU functions
**************************************************************/

// Kernel for computing the square of a vector (INPLACE)
// We actually only need z ** 2 in the computations and not z
// The square norm is also computed
__global__ void square_kernel_g(float *zsqrGPU, float *znormGPU, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while(idx < n){
        float zi = zsqrGPU[idx];
        float zsqr_i = zi * zi;
		    zsqrGPU[idx] = zi * zi;
        atomicAdd(znormGPU, zsqr_i);
		    idx += gridDim.x * blockDim.x;
	  }
}


// Device function for computing f (the secular function of interest) at a given point x
__device__ float secfunc_g(float *dGPU, float *zsqrGPU, float rho, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        sum += zsqrGPU[i] / (dGPU[i] - x);
	  }

    return rho + sum;
}


// Device function for computing f' (the prime derivative of the secular function of interest) at a given point x
__device__ float secfunc_prime_g(float *dGPU, float *zsqrGPU, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        int di = dGPU[i];
		    sum += zsqrGPU[i] / ((di - x) * (di - x));
    }

	  return sum;
}


// Device function for computing f'' (the second derivative of the secular function of interest)
__device__ float secfunc_second_g(float *dGPU, float *zsqrGPU, float x, int n){
    float sum = 0;

		for (int i = 0; i < n; i++) {
		    float di = dGPU[i];
				sum += zsqrGPU[i] / ((di - x) * (di - x) * (di - x));
		}

		return 2 * sum;
}


// Useful intermediary function, see equations (30) and (31) from Li's paper on page 13 and equation (42) on page 20
__device__ float discrimant_int_g(float a, float b, float c){

    if (a <= 0) return (a - sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a + sqrtf(a * a - 4 * b *c));
}


// Useful intermediary function, see equation (46) from Li's paper on page 21
__device__ float discrimant_ext_g(float a, float b, float c){

    if (a >= 0) return (a + sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a - sqrtf(a * a - 4 * b *c));
}


// h partition of the secular function, used for Initialization
__device__ float h_secfunc_g(float d_k, float d_kplus1, float zsqr_k, float zsqr_kplus1, float x){

    return zsqr_k / (d_k - x) + zsqr_kplus1 / (d_kplus1 - x);
}


// Initialization for interior roots (see section 4 of Li's paper - initial guesses from page 18)
__device__ float initialization_int_g(float *dGPU, float *zsqrGPU, float rho, int k, int n){

    float d_k = dGPU[k];
    float d_kplus1 = dGPU[k + 1];
    float zsqr_k = zsqrGPU[k];
    float zsqr_kplus1 = zsqrGPU[k + 1];
    float middle = (d_k + d_kplus1) / 2;
    float delta = d_kplus1 - d_k;
    float f = secfunc_g(dGPU, zsqrGPU, rho, middle, n);
    float c = f - h_secfunc_g(d_k, d_kplus1, zsqr_k, zsqr_kplus1, middle);

    if (f >= 0){
        float a = c * delta + zsqr_k + zsqr_kplus1;
        float b = zsqr_k * delta;
        return discrimant_int_g(a, b, c) + d_k;
    }

    else {
        float a = - c * delta + zsqr_k + zsqr_kplus1;
        float b = - zsqr_kplus1 * delta;
        return discrimant_int_g(a, b, c) + d_kplus1;
    }
}


// Initialization for the exterior root (see section 4 of Li's paper - initial guesses from page 18)
__device__ float initialization_ext_g(float *dGPU, float *zsqrGPU, float *znormGPU, float rho, int n){

    float d_nminus1 = dGPU[n - 1];
    float d_nminus2 = dGPU[n - 2];
    float d_n = d_nminus1 + znormGPU[0] / rho;
    float zsqr_nminus1 = zsqrGPU[n - 1];
    float zsqr_nminus2 = zsqrGPU[n - 2];
    float middle = (d_nminus1 + d_n) / 2;
    float f = secfunc_g(dGPU, zsqrGPU, rho, middle, n);
    if (f <= 0){
        float hd = h_secfunc_g(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, d_n);
        float c = f - h_secfunc_g(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        if (c <= - hd) {
            return d_n;
        }

        else {
            float delta = d_nminus1 - d_nminus2;
            float a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
            float b = - zsqr_nminus1 * delta;
            return discrimant_ext_g(a, b, c) + d_n;
        }
    }

    else {
        float delta = d_nminus1 - d_nminus2;
        float c = f - h_secfunc_g(d_nminus2, d_nminus1, zsqr_nminus2, zsqr_nminus1, middle);
        float a = - c * delta + zsqr_nminus2 + zsqr_nminus1;
        float b = - zsqr_nminus1 * delta;
        return discrimant_ext_g(a, b, c) + d_n;
    }
}


// Computation of a from the paper (page 13)
__device__ float a_gragg_g(float f, float fprime, float delta_k, float delta_kplus1){

    return (delta_k + delta_kplus1) * f - delta_k * delta_kplus1 * fprime;

}


// Computation of b from the paper (page 13)
__device__ float b_gragg_g(float f, float delta_k, float delta_kplus1){

    return delta_k * delta_kplus1 * f;
}


// Computation of c from the section Gragg of the paper (page 15)
__device__ float c_gragg_g(float f, float fprime, float fsecond, float delta_k, float delta_kplus1){

    return f - (delta_k + delta_kplus1) * fprime + delta_k * delta_kplus1 * fsecond / 2.0;

}


// Compute of the update for x (eta) for the interior roots (see section 3.1 - Iteration fomulas, pages 12 and 13)
__device__ float eta_int_g(float d_k, float d_kplus1, float f, float fprime, float fsecond, float x, int k, int n){

    float delta_k = d_k - x;
    float delta_kplus1 = d_kplus1 - x;
    float a = a_gragg_g(f, fprime, delta_k, delta_kplus1);
    float b = b_gragg_g(f, delta_k, delta_kplus1);
    float c = c_gragg_g(f, fprime, fsecond, delta_k, delta_kplus1);
    float eta = discrimant_int_g(a, b, c);
    return eta;
}

// Compute of the update of x (+eta) for the exterior root
__device__ float eta_ext_g(float d_nminus2, float d_nminus1, float f, float fprime, float fsecond, float x, int n){

    float delta_nminus2 = d_nminus2 - x;
    float delta_nminus1 = d_nminus1 - x;
    float a = a_gragg_g(f, fprime, delta_nminus2, delta_nminus1);
    float b = b_gragg_g(f, delta_nminus2, delta_nminus1);
    float c = c_gragg_g(f, fprime, fsecond, delta_nminus2, delta_nminus1);
    float eta = discrimant_ext_g(a, b, c);
    return eta;
}

// Iterate to find the k-th interior root
__device__ float find_root_int_g(float *dGPU, float *zsqrGPU, float rho, float x, int k, int n, int maxit, float epsilon, float * avloss_GPU){

    int i = 0;
    float f = secfunc_g(dGPU, zsqrGPU, rho, x, n);;
    float d_k = dGPU[k];
    float d_kplus1 = dGPU[k + 1];

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc_g(dGPU, zsqrGPU, rho, x, n);
        float fprime = secfunc_prime_g(dGPU, zsqrGPU, x, n);
        float fsecond = secfunc_second_g(dGPU, zsqrGPU, x, n);
        float eta = eta_int_g(d_k, d_kplus1, f, fprime, fsecond, x, k, n);
        x += eta;
        i ++;
    }

    // Save the loss
    atomicAdd(avloss_GPU, (float)(abs(f)/n));
    return x;
}


// Iterate to  find the last root (the exterior one)
__device__ float find_root_ext_g(float *dGPU, float *zsqrGPU, float rho, float x, int n, int maxit, float epsilon, float* avloss_GPU){

    int i = 0;
    float d_nminus2 = dGPU[n - 2];
    float d_nminus1 = dGPU[n - 1];
    float f = secfunc_g(dGPU, zsqrGPU, rho, x, n);

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc_g(dGPU, zsqrGPU, rho, x, n);
        float fprime = secfunc_prime_g(dGPU, zsqrGPU, x, n);
        float fsecond = secfunc_second_g(dGPU, zsqrGPU, x, n);
        float eta = eta_ext_g(d_nminus2, d_nminus1, f, fprime, fsecond, x, n);
        x += eta;
        i ++;
    }
      // Save the loss
    atomicAdd(avloss_GPU, (float)(abs(f)/n));
    return x;
}


// Kernel to launch and distribute the searching of roots among GPU cores
__global__ void find_roots_kernel_g(float *xstarGPU, float *x0GPU, float *dGPU, float *zsqrGPU, float *znormGPU, float rho, int n, int maxit, int epsilon, float *avloss_GPU){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // We make sure that the average loss is set to 0
    *avloss_GPU =0;

		// First core gets search of the last root (the exterior one)
    if (idx == 0){
        float x = x0GPU[n - 1];
        xstarGPU[n - 1] = find_root_ext_g(dGPU, zsqrGPU, rho, x, n, maxit, epsilon, avloss_GPU);
    }

		// Each next core searches one interval (interior interval)
    else {
        while (idx < n) {
            float x = x0GPU[idx - 1];
            xstarGPU[idx - 1] = find_root_int_g(dGPU, zsqrGPU, rho, x, idx - 1, n, maxit, epsilon, avloss_GPU);
						// in case we have not launched enough cores to cover all intervals
          	idx += gridDim.x * blockDim.x;
        }
    }
}



// Kernel to compute the initial guesses from the paper on GPU
__global__ void initialize_x0_kernel_g(float *x0GPU, float *dGPU, float *zsqrGPU, float *znormGPU, float rho, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

		// First core compute the initial guess for last root (the exterior one)
    if (idx == 0){
        x0GPU[n - 1] = initialization_ext_g(dGPU, zsqrGPU, znormGPU, rho,  n);
    }

		// Each next core compute initial guess for one interval (interior interval)
    else {
    		while (idx < n) {
        		x0GPU[idx - 1] = initialization_int_g(dGPU, zsqrGPU, rho, idx - 1, n);
        	  idx += gridDim.x * blockDim.x;
        }
    }
}

// Kernel to "wake up" the GPU
__global__ void wake_up(int *test)
{
  __shared__ int c;
  c = 3;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < 1024)
	{
		test[idx] += c;
	}
}


int main (void) {
    /****************** Access for writing ******************/
    FILE *f = fopen("result_double.csv", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f, "n;iter;niter;time_GPU;time_CPU_double;loss_GPU;loss_CPU_double\n");

    /****************** Declaration ******************/
    // Declare vectors or floats
    double *d, *z, *zsqr, *znorm, *x0, *xstar, *loss_GPU, *loss_CPU;


    // rho parameter
    double rho = 10;


    // Size of arrow matrix chosen by the user
    int n, nlow, nhigh, step, niter;
    printf("\nLowest n to test? \n");
    scanf("%d", &nlow);
    printf("\nHighest n to test? \n");
    scanf("%d", &nhigh);
    printf("\nSize of the step? \n");
    scanf("%d", &step);
    printf("\nNumber of iterations of the same n to avoid stochastic error? \n");
    scanf("%d", &niter);
    printf("\n \n******************* CHOICE OF N ******************** \n");
    printf("We compare the chosen algorithms every %d n, for n between %d and %d \n", step, nlow, nhigh);
    printf("Each test is repeated %d times \n\n", niter);
    printf("\n \n********************** TESTS *********************** \n");


    for(n=nlow; n<=nhigh; n+=step){

      //Maximum number of iterations
      int maxit = 1e4;

      //Stopping criterion
      double epsilon = 1e-6;

      // Memory allocation for data
      d = (double*)malloc(n*sizeof(double));
      z = (double*)malloc(n*sizeof(double));
      zsqr = (double*)malloc(n*sizeof(double));

      for (int iter =0; iter<niter; iter++){
        // Memory allocation for computation
    		znorm = (double*)malloc(sizeof(double));
    		x0 = (double*)malloc(n*sizeof(double));
        xstar = (double*)malloc(n*sizeof(double));
        loss_GPU = (double*)malloc(sizeof(double));
        loss_CPU = (double*)malloc(sizeof(double));

        // Create instance of class Timer
        Timer TimG, TimC;


        //Fill the vector d with linear function of n
        for (int i=0; i < n; i++){
            d[i] = 2 * n - i;
        }

        // sort the vector in ascending order
        qsort(d, n, sizeof(double), compare_function);

        // Gaussian rank 1 perturbation
        float mu_z = 5;
        float sigma_z = 1;
        gaussian_vector(z, mu_z, sigma_z, n);
        gaussian_vector(zsqr, mu_z, sigma_z, n);


        /*************************************************************************
        ********************************* CPU ************************************
        *************************************************************************/
        // Start timer CPU
        TimC.start();

        // We first compute the square and squared norm
        square_vector(zsqr, znorm, n);

        // Initialization of x0
        initialize_x0(x0, d, zsqr, znorm, rho, n);


        /***************** Root computation ****************/
        // Find roots
        find_roots(xstar, x0, d, zsqr, znorm, rho, n, maxit, epsilon, loss_CPU);

        // End timer
        TimC.add();

        // Record the performance
        *loss_GPU =0;
        fprintf(f, "%d;%d;%d;%f;%f;%f;%f\n", n, iter, niter, (double)TimG.getsum(), (float)TimC.getsum(), *loss_GPU, *loss_CPU);

        // Free memory used for computation on CPU
        free(znorm);
        free(xstar);
        free(loss_CPU);
        free(loss_GPU);
      }

    printf("%d has been tested\n", n);
    // Free memory used to store data on CPU
    free(d);
    free(z);
    free(zsqr);
    }
    printf("\n \n");

    // We close the access to the file
    fclose(f);
}

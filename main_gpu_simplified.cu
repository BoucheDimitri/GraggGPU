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
if (*x < *y) return 1;
else if (*x > *y) return -1; 
return 0;
}


// Generate gaussian vector using Box Muller
void gaussian_vector(float *v, float mu, float sigma, int n){
	for (int i = 0; i<n; i++){
		float u1 = (float)rand()/(float)(RAND_MAX);
		float u2 = (float)rand()/(float)(RAND_MAX);
		v[i] = sigma * (sqrtf( -2 * logf(u1)) * cosf(2 * M_PI * u2)) + mu;
	} 
}


//Function to print a small square matrix of floats on host
void print_matrix(float *c, int n) {

	for (int i=0; i<n; i++){

    		for(int j=0; j<n; j++) {

         		printf("%f     ", c[n * i + j]);
        	}

    		printf("\n");
 	}
}

//Function to print a small vector of floats on host
void print_vector(float *c, int m, int n) {

	for (int i=0; i<m; i++){

		printf("%f     ", c[i]);

    		printf("\n");
 	}
}



// Fill c with arrow matrix generated from vectors a and b
// Not very useful actually for our problem
void generate_arrow(float *a, float *b, float *c, float gamma, int n) {

	int j = 0;

	// Fill the arrow
	for (int i=0; i<n; i ++){

		if (i<n-1) {

		// Iterate over the last column of c
		c[n - 1 + i*n] = b[i];

		// Iterate over the last row of c
		c[n * (n-1) + i] = b[i];

		// Iterate over the diagonal of c
		c[n*i + j] = a[i];
		j ++;

		}
	}

	// Fill last element of diagonal with gamma
	c[(n-1) * (n-1)] = gamma;
}


// Kernel for computing the square of a vector
// We actually only need the square of b in the computations
// Thus it is better to compute it once and for all
// We also collect the squared norm of the vector
__global__ void square_kernel(float *bGPU, float *bsqrGPU, float *bnorm, int n){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	while(idx < n){
		bsqrGPU[idx] = bGPU[idx] * bGPU[idx];
		atomicAdd(bnorm, bsqrGPU[idx]);
		idx += gridDim.x * blockDim.x;
	}
}



// Device function for computing f (the spectral function) at a given point x
__device__ float spectral_func(float *aGPU,
			       float *bsqrGPU,
			       float x,
			       float gamma,
			       int n) {

	float sum = 0;

	for (int i=0; i<n-1; i++){
		sum += bsqrGPU[i] / (aGPU[i] - x);
	}

	return x - gamma + sum;
}


// Device function for computing f' (the prime derivative of the spectral function) at a given point x
__device__ float spectral_func_prime(float *aGPU,
			       	     float *bsqrGPU,
			             float x,
			             int n) {

	float sum = 0;

	for (int i=0; i<n-1; i++){

		int ai_local = aGPU[i];
		sum += bsqrGPU[i] / ((ai_local - x) * (ai_local - x));
	}

	return 1 + sum;
}


// Device function to compute the interior versions of sigma
__device__ float interior_sigma(float *aGPU,
		       		float *bsqrGPU,
		                float x,
		                float gamma,
				int k,
		                int n) {

	float sum = 0;

	//Use the registers
	float ak_local = aGPU[k];
	float ak_minus1_local = aGPU[k-1];

	for (int i=0; i<n-1; i++) {

		//Use the registers
		float ai_local = aGPU[i];

		float num = bsqrGPU[i] * (ai_local - ak_minus1_local) * (ai_local - ak_local);

		float deno = (ai_local - x) * (ai_local - x)
		        * (ai_local - x);

		sum +=  num / deno;
	}

	float term1 = 3 * x - gamma - ak_local - ak_minus1_local;

	return term1 + sum;
}


// Interior version for computation of alpha
__device__ float interior_alpha(float sigma, float x, float ak, float ak_minus1){

	return sigma / ((ak_minus1 - x) * (x - ak));
}


__device__ float first_alpha(float x, float *aGPU, float *bsqrGPU, int n){

	float sum = 0;
	float a0 = aGPU[0];

	for (int i=1; i<n-1; i++) {
		float ai = aGPU[i]; 
		sum += (bsqrGPU[i]*(a0-ai))/((x-ai)*(x-ai)*(x-ai));
	}

	return -(1+sum)/(x-a0);
	
}


__device__ float last_alpha(float x, float *aGPU, float *bsqrGPU, int n){

	float sum = 0;
	float a_nminus2 = aGPU[n-2];

	for (int i=0; i<n-2; i++) {
		float ai = aGPU[i]; 
		sum += (bsqrGPU[i]*(a_nminus2 - ai)) / ((x - ai)*(x - ai)*(x - ai));
	}

	return -(1 + sum)/(x - a_nminus2);
}



// Interior version for computation of beta
__device__ float interior_beta(float fprime, float f, float x, float ak, float ak_minus1){

	float fac = (1 / (ak_minus1 - x) + 1 / (ak - x));
	return fprime - fac * f;

}


// Computation for exterior betas
// if a = a[0], the first, if a = a[n-2], the last
__device__ float exterior_beta(float fprime, float f, float x, float a){

	return fprime + f/(x - a);

}



// Computation of the update (delta) on device
__device__ float interior_delta(float f, float alpha, float beta){

	float term1 = 2 * f / beta;
	float term2 = 2 * alpha / beta;
	float deno = 1 + sqrtf(1 + term1 * term2);
	return term1 / deno;
}


// device function to find the zero within the interval (a[k], a[k-1])
__device__ float interior_zero_finder(float *aGPU,
			   	      float *bsqrGPU,
			              float gamma,
			   	      float x,
			   	      int k,
			   	      int n,
			   	      int maxit,
			   	      float epsilon){

	int i = 0;
	// To guarantee entry in the loop
	float f = 2 * sqrtf(epsilon);
	while ((i < maxit) && (f*f > epsilon)){
		// Computation of sigma(x), solution of system (5) in page 7 (12 in the pdf) of the article
		float sig = interior_sigma(aGPU, bsqrGPU, x, gamma, k, n);
		float ak_local = aGPU[k];
		float ak_minus1_local = aGPU[k - 1];
		// Computation of alpha(x), see definition (7) of the article in page 8 (13 in the pdf)
		float alpha = interior_alpha(sig, x, ak_local, ak_minus1_local);
		// Computation of spectral_func(x)
		f = spectral_func(aGPU, bsqrGPU, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(aGPU, bsqrGPU, x, n);
		// Computation of beta(x), see definition (8) of the article in page 8 (13 in the pdf)
		float beta = interior_beta(fprime, f, x, ak_local, ak_minus1_local);
		// Computation of delta(x), see definition (9) of the article in page 8 (13 in the pdf)
		float delta = interior_delta(f, alpha, beta);
		// Update of x
		x -= delta;
		i ++;
	}
	// We print a few controls to check the quality of the root obtained
	if (k%(int)(n/10) ==0){
		printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n"
		, k, x, f*f, i);
	}
	return x;
}


__device__ float first_zero_finder(float *aGPU,
			   	   float *bsqrGPU,
			   	   float gamma,
			           float x,
			   	   int n,
				   int maxit,
				   float epsilon){

	int i = 0;
	float a0 = aGPU[0];
	// To guarantee entry in the loop
	float f = 2 * sqrtf(epsilon);
	while ((i < maxit) && (f*f > epsilon)){

		// Computation of alpha(x)
		float alpha = first_alpha(x, aGPU, bsqrGPU, n);
		// Computation of spectral_func(x)
		f = spectral_func(aGPU, bsqrGPU, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(aGPU, bsqrGPU, x, n);
		// Computation of beta(x)
		float beta = exterior_beta(fprime, f, x, a0);
		// Computation of delta(x)
		float delta = interior_delta(f, alpha, beta);
		// Update of x
		x -= delta;
		i ++;
	}
	
	// We print a control to check the quality of the root obtained
	printf("\n");
		printf("********************* CONTROLS ********************* \n");
		printf("We print the first, the last and 10 %% of the interior eigenvalues as a check \n");
	
	printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n"
	, 0, x, f*f, i);

	return x;
}



__device__ float last_zero_finder(float *aGPU,
			   float *bsqrGPU,
			   float gamma,
			   float x,
			   int n,
			   int maxit,
			   float epsilon){

	int i = 0;
	float a_nminus2 = aGPU[n-2];
	// To guarantee entry in the loop
	float f = 2 * sqrtf(epsilon);
	while ((i < maxit) && (f*f > epsilon)){
		// Computation of alpha(x)
		float alpha = last_alpha(x, aGPU, bsqrGPU, n);
		// Computation of spectral_func(x)
		f = spectral_func(aGPU, bsqrGPU, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(aGPU, bsqrGPU, x, n);
		// Computation of beta(x)
		float beta = exterior_beta(fprime, f, x, a_nminus2);
		// Computation of delta(x)
		float delta = interior_delta(f, alpha, beta);
		// Update of x
		x -= delta;
		i ++;
	}
	// We print a control to check the quality of the root obtained
	printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n"
	, n-1, x, f*f, i);
	return x;
}



// Kernel for paralell initialization of x0s on the GPU
__global__ void initialize_x0_kernel(float *aGPU, float *x0_vecGPU, float *bnorm, float gamma, int n){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Initial x0 for first interval
	if (idx == 0) {
		float a0_local = aGPU[0]; 
		float bnorm0_local = bnorm[0];
		float term1 = (gamma - a0_local)/2;
		if (gamma > a0_local){
			x0_vecGPU[0] = a0_local + term1 +sqrtf(term1 * term1 + bnorm0_local);
		}
		else {
			x0_vecGPU[0] = a0_local + bnorm0_local / (-term1 +sqrtf(term1 * term1 + bnorm0_local));
		}
	}

	// Initial x0 for last interval
	else if (idx == 1) {
		float aGPUnminus2_local = aGPU[n-2];
		float bnorm0_local = bnorm[0];
		float term2 = (gamma - aGPUnminus2_local)/2;
		float a0_local = aGPU[0]; 
		float term1 = (gamma - a0_local)/2;
		if (gamma > aGPUnminus2_local){
			x0_vecGPU[n-1] = aGPUnminus2_local - term2 - sqrtf(term1 * term1 + bnorm0_local);
			//x0_vecGPU[n-1] = aGPUnminus2_local - term2 -sqrtf(term2 * term2 + bnorm0_local);
		}
		else {
			x0_vecGPU[n-1] = aGPUnminus2_local - bnorm0_local / (-term2 + sqrtf(term2 * term2 + bnorm0_local));
		}
	}

	// Interior intervals	
	else {
		while (idx < n) {
			x0_vecGPU[idx - 1] = (aGPU[idx - 2] + aGPU[idx - 1]) / 2 ;
			idx += gridDim.x * blockDim.x;
		}	
	}
}



// Kernel to find the zeros (only the interior ones for now)
__global__ void find_zeros_kernel(float *aGPU,
				  float *bsqrGPU,
				  float *bnorm,
				  float *x0_vecGPU,
				  float *xstar_vecGPU,
				  float gamma,
				  int n,
				  int maxit,
				  float epsilon) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	/***************** Root Computation *****************/
	// First eigenvalue
	if (idx == 0){
		// Initial value
		float x = x0_vecGPU[0];
		// First core gets first interval and finds the unique zero within
		xstar_vecGPU[0] = first_zero_finder(aGPU, bsqrGPU, gamma, x, n, maxit, epsilon) ;
		// In case n - 2 > gridDim.x * blockDim.x : USELESS
		//idx += gridDim.x * blockDim.x;
	}
	
	// Last eigenvalue
	else if (idx == 1) {
		// Initial value
		float x = x0_vecGPU[n-1];
		// Second core gets last interval and finds the unique zero within
		xstar_vecGPU[n-1] = last_zero_finder(aGPU, bsqrGPU, gamma, x, n, maxit, epsilon) ;
		// In case n - 2 > gridDim.x * blockDim.x : USELESS
		//idx += gridDim.x * blockDim.x;

	}

	// All interior eigenvalues
	else {
		while (idx < n) {
		// Initial value
		float x = x0_vecGPU[idx - 1];
		// Each core gets an interior interval and finds the unique zero within
		xstar_vecGPU[idx - 1] = interior_zero_finder(aGPU, bsqrGPU, gamma, x, idx - 1, n, maxit, epsilon);
		idx += gridDim.x * blockDim.x;
		}
	}
}


int main (void) {

	/****************** Declaration ******************/
	// Declare vectors or floats
	float *a, *b, *xstar_vec, *c; //*x0_vec;


	// Gamma
	float gamma = 10;


	// Size of arrow matrix chosen by the user
	//int n= 10;
	int n; 
	printf("\nWhich n (number of roots for the function) do you want? \n");
	scanf("%d", &n);
	printf("\n \n******************* CHOICE OF N ******************** \n");
	printf("n = %d\n", n);

	//Maximum number of iterations
	int maxit = 1e5;


	//Stopping criterion
	float epsilon = 0.000001;


	// Memory allocation
	a = (float*)malloc((n-1)*sizeof(float));
	b = (float*)malloc((n-1)*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));
	//x0_vec = (float*)malloc(n*sizeof(float));
	xstar_vec = (float*)malloc(n*sizeof(float));


	// Create instance of class Timer
	Timer Tim;


	// Fill the vectors a and b (arbitrarily for now)
	//for (int i=0; i<n-1; i++){
	//	a[i] = 0.5 * n - 0.1 * i;
	//}

	// Fill a as a vector of gaussian of mean mu and std sigma 
	float mu = 50;
	float sigma = 1;
	gaussian_vector(a, mu, sigma, n-1);
	// We sort by descending order then
	qsort(a, n-1, sizeof(float), compare_function);
	

	for (int i=0; i<n-1; i++){
		b[i] = 1;
	}

	// Start timer
	Tim.start();

	/***************** GPU memory alloc *****************/

	// Declare vectors on GPU
	float *aGPU, *bGPU, *bsqrGPU, *bnorm, *x0_vecGPU, *xstar_vecGPU;

	// Create memory space for vectors on GPU
	cudaMalloc(&aGPU, (n-1)*sizeof(float));
	cudaMalloc(&bGPU, (n-1)*sizeof(float));
	cudaMalloc(&bsqrGPU, (n-1)*sizeof(float));
	cudaMalloc(&bnorm, (1)*sizeof(float));
	// The initial values
	cudaMalloc(&x0_vecGPU, n*sizeof(float));
	// Container for the results
	cudaMalloc(&xstar_vecGPU, n*sizeof(float));


  	/***************** Transfer on GPU *****************/


	// Transfers on GPU
	cudaMemcpy(aGPU, a, (n-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bGPU, b, (n-1)*sizeof(float), cudaMemcpyHostToDevice);


	// We first compute the square and squared norm
	square_kernel <<<1024, 512>>> (bGPU, bsqrGPU, bnorm, n);


	// Initialization of x0 on GPU
	initialize_x0_kernel<<<1024, 512>>> (aGPU, x0_vecGPU, bnorm, gamma, n);
	


	/***************** Root computation ****************/
	// Find interior zeros on GPU
	// (it includes initilisation)
	find_zeros_kernel<<<1024, 512>>> (aGPU,
					  bsqrGPU,
					  bnorm,
					  x0_vecGPU,
					  xstar_vecGPU,
					  gamma,
					  n,
					  maxit,
					  epsilon);


	// Transfer results on CPU to print it
	cudaMemcpy(xstar_vec, xstar_vecGPU, n*sizeof(float), cudaMemcpyDeviceToHost);

	//cudaMemcpy(x0_vec, x0_vecGPU, n*sizeof(float), cudaMemcpyDeviceToHost);


	// End timer
	Tim.add();


	// Print the first zeros
	// Number of roots to display
	int m = 10;
	printf("\n********************* RESULTS ********************** \n");
	printf("The first %i greater resulting roots (eigen values) are : \n", m);
	print_vector(xstar_vec, m, n);


	// Print how long it took
	printf("GPU timer for root finding (CPU-GPU and GPU-CPU transfers included) : %f s\n\n",
		(float)Tim.getsum());

	//print_vector(x0_vec, 10, n);



	// Free memory on GPU
	cudaFree(aGPU);
	cudaFree(bGPU);
	cudaFree(bsqrGPU);
	cudaFree(x0_vecGPU);
	cudaFree(xstar_vecGPU);

	//printf("a"); 
	// Free memory on CPU
	free(a);
	free(b);
	printf("c"); 
	free(c);
	//free(x0_vec);
	printf("xstart"); 
	//free(xstar_vec);

}

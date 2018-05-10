#include <stdio.h>

/**************************************************************
The code in time.h is a part of a course on cuda taught by its authors:
Lokman A. Abbas-Turki
**************************************************************/
#include "timer.h"


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

// Compute the initial values
__device__ void x0_initialisation(float *aGPU, float *x0_vecGPU, float *bnorm, float gamma, int n){

	// First eigenvalue
	float term1 = (gamma-aGPU[0])/2;
	if (gamma > aGPU[0]){
		x0_vecGPU[0] = aGPU[0] + term1 +sqrtf(term1 * term1 + bnorm[0]);
	}
	else {
		x0_vecGPU[0] = aGPU[0] + bnorm[0] / (-term1 +sqrtf(term1 * term1 + bnorm[0]));
	}

	// Interior eigenvalues
	// We take the middle of the intervals as initial value
	//(as advised in the paper at the beginning of  page 8 (13 of the pdf)
	for (int i=1; i<n-1; i++){
		x0_vecGPU[i] = (aGPU[i-1] + aGPU[i]) / 2 ;
	}

	// Last eigenvalue
	float term2 = (gamma-aGPU[n-2])/2;
	if (gamma > aGPU[n-2]){
		x0_vecGPU[n-1] = aGPU[n-2] - term2 -sqrtf(term1 * term1 + bnorm[0]);
	}
	else {
		x0_vecGPU[n-1] = (float)aGPU[n-2] - (float)bnorm[0] / (-(float)term2 +(float)sqrtf((float)term2 * (float)term2 + (float)bnorm[0]));
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


// Exterior version for computation of alpha
__device__ float exterior_alpha(float x, float *aGPU, int n, int k, float *bsqrGPU){

	float sum = 0;

	if (k==0){
		float a0 = aGPU[0];
		for (int i=1; i<n-1; i++) {
			sum += (bsqrGPU[i]*(a0-aGPU[i]))/((x-aGPU[i])*(x-aGPU[i])*(x-aGPU[i]));
		}
		return -(1+sum)/(x-a0);
	}

	else{
		float a_nminus2 = aGPU[n-2];
		for (int i=0; i<n-2; i++) {
			sum += (bsqrGPU[i]*(a_nminus2-aGPU[i]))/((x-aGPU[i])*(x-aGPU[i])*(x-aGPU[i]));
		}
		return -(1+sum)/(x-a_nminus2);
	}
}


// Interior version for computation of beta
__device__ float interior_beta(float fprime, float f, float x, float ak, float ak_minus1){

	float fac = (1 / (ak_minus1 - x) + 1 / (ak - x));
	return fprime - fac * f;

}

// Exterior version for computation of beta
__device__ float exterior_beta(float fprime, float f, int k, float x, float a0, float a_nminus2){
	if (k==0){
		return fprime + f/(x-a0);
	}

	else{
		return fprime + f/(x-a_nminus2);
	}
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


__device__ float exterior_zero_finder(float *aGPU,
			   float *bsqrGPU,
			   float gamma,
			   float x,
				 int k,
			   int n,
			   int maxit,
			   float epsilon){

	int i = 0;
	float a0 = aGPU[0];
	float a_nminus2 = aGPU[n-2];
	// To guarantee entry in the loop
	float f = 2 * sqrtf(epsilon);
	while ((i < maxit) && (f*f > epsilon)){

		// Computation of alpha(x)
		float alpha = exterior_alpha(x, aGPU, n, k, bsqrGPU);
		// Computation of spectral_func(x)
		f = spectral_func(aGPU, bsqrGPU, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(aGPU, bsqrGPU, x, n);
		// Computation of beta(x)
		float beta = exterior_beta(fprime, f, k, x, a0, a_nminus2);
		// Computation of delta(x)
		float delta = interior_delta(f, alpha, beta);
		// Update of x
		x -= delta;
		i ++;
	}
	// We print a control to check the quality of the root obtained
	if (k==0){
		printf("\n");
		printf("********************* CONTROLS ********************* \n");
		printf("We print the first, the last and 10 %% of the interior eigenvalues as a check \n");
	}
	printf("eigenvalue %d: %f, with spectral function %f after %d iterations \n"
	, k, x, f*f, i);
	return x;
}


// Kernal to find the zeros (only the interior ones for now)
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

	/****************** Initialisation ******************/
	x0_initialisation(aGPU, x0_vecGPU, bnorm, gamma, n);
	__syncthreads();

	/***************** Root Computation *****************/
	// First eigenvalue
	if (idx == 0){
		// Initial value
		float x = x0_vecGPU[idx];
		// Each core gets an interior interval and finds the unique zero within
		xstar_vecGPU[idx] = exterior_zero_finder(aGPU, bsqrGPU, gamma, x, idx, n, maxit, epsilon);
		// In case n - 2 > gridDim.x * blockDim.x
		idx += gridDim.x * blockDim.x;
	}


	// All interior eigenvalues
	if ((idx>0)&&(idx<n-1)){
		// Initial value
		float x = x0_vecGPU[idx];
		// Each core gets an interior interval and finds the unique zero within
		xstar_vecGPU[idx] = interior_zero_finder(aGPU, bsqrGPU, gamma, x, idx, n, maxit, epsilon);
		// In case n - 2 > gridDim.x * blockDim.x
		idx += gridDim.x * blockDim.x;
	}


	// Last eigenvalue
	if (idx == n-1){
		// Initial value
		float x = x0_vecGPU[idx];
		// Each core gets an interior interval and finds the unique zero within
		xstar_vecGPU[idx] = exterior_zero_finder(aGPU, bsqrGPU, gamma, x, idx, n, maxit, epsilon);
		// In case n - 2 > gridDim.x * blockDim.x
		idx += gridDim.x * blockDim.x;
	}

}


// KERNEL FOR TESTING, TO BE REMOVED, IGNORE
__global__ void test_all_kernel(float *aGPU,
				float *bsqrGPU,
				float *yvecGPU,
				float *xvecGPU,
				float gamma,
				int n) {


	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	while(idx < n-2){
		float x_local = xvecGPU[idx + 1];
		float sig = interior_sigma(aGPU, bsqrGPU, x_local, gamma, idx + 1, n);
		float ak_local = aGPU[idx + 1];
		float ak_minus1_local = aGPU[idx];
		float alpha = interior_alpha(sig, x_local, ak_local, ak_minus1_local);
		float f = spectral_func(aGPU, bsqrGPU, x_local, gamma, n);
		float fprime = spectral_func_prime(aGPU, bsqrGPU, x_local, n);
		float beta = interior_beta(fprime, f, x_local, ak_local, ak_minus1_local);
		float delta = interior_delta(f, alpha, beta);
		yvecGPU[idx + 1] = delta;
		idx += gridDim.x * blockDim.x;
	}
}


int main (void) {

	/****************** Declaration ******************/
	// Declare vectors or floats
	float *a, *b, *xstar_vec, *c;


	// Gamma
	float gamma = 1;


	// Size of arrow matrix chosen by the user
	int n= 10;
	printf("\nWhich n (number of roots for the function) do you want? \n");
	scanf("%d", &n);
	printf("\n \n******************* CHOICE OF N ******************** \n");
	printf("n = %d\n", n);

	//Maximum number of iterations
	int maxit = 1e4;


	//Stopping criterion
	float epsilon = 0.000001;


	// Memory allocation
	a = (float*)malloc((n-1)*sizeof(float));
	b = (float*)malloc((n-1)*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));
	xstar_vec = (float*)malloc(n*sizeof(float));


	// Create instance of class Timer
	Timer Tim;


	// Fill the vectors a and b (arbitrarily for now)
	for (int i=0; i<n; i++){
		a[i] = 2 * n - i;
	}

	for (int i=0; i<n-1; i++){
		b[i] = n - i;
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
	// We first compute the square and squared norm
	square_kernel <<<1024, 512>>> (bGPU, bsqrGPU, bnorm, n);

	// Transfers on GPU
	cudaMemcpy(aGPU, a, (n-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bGPU, b, (n-1)*sizeof(float), cudaMemcpyHostToDevice);


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



	// Free memory on GPU
	cudaFree(aGPU);
	cudaFree(bGPU);
	cudaFree(bsqrGPU);
	cudaFree(x0_vecGPU);
	cudaFree(xstar_vecGPU);


	// Free memory on CPU
	free(a);
	free(b);
	free(c);
	free(xstar_vec);

}

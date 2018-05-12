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

void positive_part_vec(float *v, int n){
	for (int i = 0; i<n; i++){

		v[i] = max((float)0, v[i]);
	}
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

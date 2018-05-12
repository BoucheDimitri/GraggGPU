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
void gaussian_vector(float *v, float mu, float sigma, int n){*

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
// We actually only need z ** 2 in the computations and not z
__global__ void square_kernel(float *zGPU, float *zsqrGPU, int n){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while(idx < n){
        float zi = zGPU[idx];
		    zsqrGPU[idx] = zi * zi;
		    idx += gridDim.x * blockDim.x;
	  }
}


// Device function for computing f (the spectral function) at a given point x
__device__ float secfunc(float *dGPU, float *zsqrGPU, float x, float rho, int n) {

    float sum = 0;
    for (int i=0; i<n-1; i++){
        sum += zsqrGPU[i] / (dGPU[i] - x);
	  }

    return x - rho + sum;
}


// Device function for computing f' (the prime derivative of the spectral function) at a given point x
__device__ float secfunc_prime(float *dGPU, float *zsqrGPU, float x, int n) {

    float sum = 0;
    for (int i=0; i<n-1; i++){
        int di = dGPU[i];
		    sum += zsqrGPU[i] / ((di - x) * (di - x));
    }

	  return 1 + sum;
}


// Device function for computing f'' (the second derivative of the spectral function)
__device__ float secfunc_second(float *dGPU, float *zsqrGPU, float x, int n){
    float sum = 0;

		for (int i = 0; i < count; i++) {
		    float di = dGPU[i];
				sum += zsqrGPU[i] / ((di - x) * (di -x) * (di -x));
		}

		return 2 * sum;
}

//
__device__ float discrimant_int(float a, float b, float c){

    if (a <= 0) return (a - sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a + sqrtf(a * a - 4 * b *c));
}


__device__ float discrimant_ext(float a, float b, float c){
    if (a >= 0) return (a + sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a - sqrtf(a * a - 4 * b *c));
}

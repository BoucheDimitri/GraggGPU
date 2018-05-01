#include <stdio.h>


//Function to print a small square matrix of floats
void print_matrix(float *c, int n) {

	for (int i=0; i<n; i++){

    		for(int j=0; j<n; j++) {

         		printf("%f     ", c[n * i + j]);
        	}

    		printf("\n");
 	}	
}



// Fill c with arrow matrix generated from vectors a and b
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
// The square of b is needed during several computations 
// for all subproblems, so better to compute it once and for all
__global__ void square_kernel(float *bGPU, float *bsqrGPU, int n){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < n){
	bsqrGPU[idx] = bGPU[idx] * bGPU[idx];
	idx += gridDim.x * blockDim.x;
	}
}











int main (void) {

	// Declare vectors
	float *a, *b, *c;

	// Size of arrow matrix
	int n = 10;
	
	// Declare and reference gamma
	float gamma = 1;
	
	// Memory allocation
	a = (float*)malloc((n-1)*sizeof(int));
	b = (float*)malloc((n-1)*sizeof(int));
	c = (float*)malloc(n*n*sizeof(int));

	// Fill the vectors
	for (int i=0; i<n; i++){
		a[i] = 20 - i;
	}

	for (int i=0; i<n-1; i++){
		b[i] = 10 - i;
	}

	// Fill c with arrow matrix generated from a and b
	generate_arrow(a, b, c, n);

	// Print c
	print_matrix(c, n);

	
	// Declare vectors on GPU
	int *aGPU, *bGPU, *cGPU;

	// Create memory space for vectors on GPU
	cudaMalloc(&aGPU, (n-1)*sizeof(float));
	cudaMalloc(&bGPU, (n-1)*sizeof(float));
	
	




	// Free memory on GPU
	cudaFree(aGPU);
	cudaFree(aGPU);


	// Free memory on CPU
	free(a);
	free(b);
	free(c);
	
}


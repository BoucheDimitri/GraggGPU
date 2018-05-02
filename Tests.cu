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

//Function to print a small vector of floats
void print_vector(float *c, int n) {

	for (int i=0; i<n; i++){

		printf("%f     ", c[i]);

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
	float *a, *b, *bsqr, *c;

	// Size of arrow matrix
	int n = 10;
	
	// Declare and reference gamma
	float gamma = 1;
	
	// Memory allocation
	a = (float*)malloc((n-1)*sizeof(float));
	b = (float*)malloc((n-1)*sizeof(float));
	bsqr = (float*)malloc((n-1)*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));

	// Fill the vectors
	for (int i=0; i<n; i++){
		a[i] = 20 - i;
	}

	for (int i=0; i<n-1; i++){
		b[i] = 10 - i;
	}

	// Fill c with arrow matrix generated from a and b
	generate_arrow(a, b, c, gamma, n);

	// Print c
	//print_matrix(c, n);

	
	// Declare vectors on GPU
	float *aGPU, *bGPU, *bsqrGPU;

	// Create memory space for vectors on GPU
	cudaMalloc(&aGPU, (n-1)*sizeof(float));
	cudaMalloc(&bGPU, (n-1)*sizeof(float));
	cudaMalloc(&bsqrGPU, (n-1)*sizeof(float));

	// Transfer on GPU
	cudaMemcpy(aGPU, a, (n-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bGPU, b, (n-1)*sizeof(float), cudaMemcpyHostToDevice);

	//Test square kernel
	square_kernel <<<1024, 512>>> (bGPU, bsqrGPU, n);

	cudaMemcpy(bsqr, bsqrGPU, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
	
	print_vector(bsqr, n-1);
	


	// Free memory on GPU
	cudaFree(aGPU);
	cudaFree(bGPU);
	cudaFree(bsqrGPU);


	// Free memory on CPU
	free(a);
	free(b);
	free(bsqr);
	free(c);
	
}


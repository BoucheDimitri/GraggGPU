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
// for all subproblems, so better to compute it once and for all at the beginning
__global__ void square_kernel(float *bGPU, float *bsqrGPU, int n){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while(idx < n){
	bsqrGPU[idx] = bGPU[idx] * bGPU[idx];
	idx += gridDim.x * blockDim.x;
	}
}



// Device function for computing f(*xGPU)
__device__ float spectral_func(float *aGPU, 
			       float *bsqrGPU, 
			       float *xGPU, 
			       float *gammaGPU, 
			       int n) {
	
	float sum = 0;

	for (int i=0; i<n-1; i++){
		sum += bsqrGPU[i] / (aGPU[i] - *xGPU);
	}
	
	return *xGPU - *gammaGPU + sum;
}


// Kernel associated with spectral_func device function
__global__ void spectral_func_kernel(float *aGPU, 
				     float *bsqrGPU, 
				     float *yvecGPU, 
				     float *xvecGPU, 
				     float *gammaGPU, 
				     int n) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	while(idx < n){

	yvecGPU[idx] = spectral_func(aGPU, bsqrGPU, &(xvecGPU[idx]), gammaGPU, n);
	idx += gridDim.x * blockDim.x;

	}

}





int main (void) {

	// Declare vectors
	float *a, *b, *bsqr, *xvec, *yvec, *c, *gamma;

	// Size of arrow matrix
	int n = 10;
	
	// Memory allocation
	a = (float*)malloc((n-1)*sizeof(float));
	b = (float*)malloc((n-1)*sizeof(float));
	bsqr = (float*)malloc((n-1)*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));
	xvec = (float*)malloc(n*sizeof(float));
	yvec = (float*)malloc(n*sizeof(float));
	gamma = (float*)malloc(sizeof(float));
	


	// Fill the vectors
	for (int i=0; i<n; i++){
		a[i] = 20 - i;
	}

	for (int i=0; i<n-1; i++){
		b[i] = 10 - i;
	}

	// We take the middle of the intervals
	for (int i=1; i<n-1; i++){
		xvec[i] = (a[i-1] + a[i]) / 2 ;
	}
	
	//Arbitrary filling of the edges values
	xvec[0] = a[0] + 5;
	xvec[n-1] = a[n-2] - 5; 

	//Fill gamma 
	*gamma = 1;


	// Fill c with arrow matrix generated from a and b
	generate_arrow(a, b, c, *gamma, n);

	// Print c
	print_matrix(c, n);

	
	// Declare vectors on GPU
	float *aGPU, *bGPU, *bsqrGPU, *xvecGPU, *yvecGPU, *gammaGPU;

	// Create memory space for vectors on GPU
	cudaMalloc(&aGPU, (n-1)*sizeof(float));
	cudaMalloc(&bGPU, (n-1)*sizeof(float));
	cudaMalloc(&bsqrGPU, (n-1)*sizeof(float));
	cudaMalloc(&xvecGPU, n*sizeof(float));
	cudaMalloc(&yvecGPU, n*sizeof(float));
	cudaMalloc(&gammaGPU, sizeof(float));
	

	// Transfers on GPU
	cudaMemcpy(aGPU, a, (n-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bGPU, b, (n-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(xvecGPU, xvec, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gammaGPU, gamma, sizeof(float), cudaMemcpyHostToDevice);


	//Compute square of b on GPU
	square_kernel <<<1024, 512>>> (bGPU, bsqrGPU, n);

	// Transfer on CPU and print to check result
	//cudaMemcpy(bsqr, bsqrGPU, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
	//print_vector(bsqr, n-1);

	//Compute spectral function on GPU
	spectral_func_kernel <<<1024, 512>>> (aGPU, bsqrGPU, yvecGPU, xvecGPU, gammaGPU, n);

	// Transfer spectral function results on CPU to print it
	cudaMemcpy(yvec, yvecGPU, n*sizeof(float), cudaMemcpyDeviceToHost);
	print_vector(yvec, n);


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

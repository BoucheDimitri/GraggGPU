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



// Device function for computing f at a given point
__device__ float spectral_func(float *aGPU, 
			       float *bsqrGPU, 
			       float x, 
			       float *gammaGPU, 
			       int n) {
	
	float sum = 0;

	for (int i=0; i<n-1; i++){
		sum += bsqrGPU[i] / (aGPU[i] - x);
	}
	
	return x - *gammaGPU + sum;
}


// Device function for computing f' at a given point
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





// Kernel associated with spectral_func device function
//__global__ void spectral_func_kernel(float *aGPU, 
//				     float *bsqrGPU, 
//				     float *yvecGPU, 
//				     float *xvecGPU, 
//				     float *gammaGPU, 
//				     int n) {
//
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	while(idx < n){
//
//	yvecGPU[idx] = spectral_func(aGPU, bsqrGPU, &(xvecGPU[idx]), gammaGPU, n);
//	idx += gridDim.x * blockDim.x;
//
//	}
//}



// Device function to compute the interior versions of sigma
__device__ float interior_sigma(float *aGPU, 
		       		float *bsqrGPU, 
		                float x, 
		                float *gammaGPU, 
				int k,
		                int n) {
	
	float sum = 0;

	//Use the registers
	float ak_local = aGPU[k];
	float ak_minus1_local = aGPU[k-1]; 

	for (int i=0; i<n-1; i++) {
		
		//Use of registers
		float ai_local = aGPU[i];
		
		float num = bsqrGPU[i] * (ai_local - ak_minus1_local) * (ai_local - ak_local);
		
		float deno = (ai_local - x) * (ai_local - x) 
		        * (ai_local - x);
		
		sum +=  num / deno;
	}

	float term1 = 3 * x - *gammaGPU - ak_local - ak_minus1_local;

	return term1 + sum;
}




// Kernel just for testing sigma interior function
// We can certainly do better in terms of paralellisation
// Also here we calculate also sigma for the two limiting intervals : 
//(a[0], +inf) and (-inf, a[n-1]), a different formula should be used
//__global__ void sigma_interior_kernel(float *aGPU, 
//				      float *bsqrGPU, 
//				      float *yvecGPU, 
//				      float *xvecGPU, 
//				      float *gammaGPU, 
//				      int n) {
//
//
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	while(idx < n){

//	yvecGPU[idx] =  interior_sigma(aGPU, bsqrGPU, &(xvecGPU[idx]), gammaGPU, n, idx);
//	idx += gridDim.x * blockDim.x;
//
//	}
//}



// Interior version for computation of alpha
//__device__ float interior_alpha(float *sigma, float *x, float *ak, float *ak_minus1){

//	return *sigma / (*ak_minus1 - *x) * (*ak - *x);
//}


// Interior version for computation of alpha
__device__ float interior_alpha(float sigma, float x, float ak, float ak_minus1){

	return sigma / ((ak_minus1 - x) * (ak - x));
}



// Interior version for computation of beta
__device__ float interior_beta(float fprime, float f, float x, float ak, float ak_minus1){

	float fac = (1 / (ak_minus1 - x) + 1 / (ak - x)); 
	return fprime - fac * f; 
	
}


// Square root device function
__device__ float square_root(float x){

	return expf(0.5 * logf(x));
}


// Computation of the update (delta)
__device__ float interior_delta(float *f, float *alpha, float *beta){


	float term1 = 2 * (*f) / (*beta);
	float term2 = 2 * (*alpha) / (*beta);
	float deno = 1 + square_root(1 + term1 * term2);
	return term1 / deno; 
}



__global__ void test_all_kernel(float *aGPU, 
				float *bsqrGPU, 
				float *yvecGPU, 
				float *xvecGPU, 
				float *gammaGPU, 
				int n) {


	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// IMPORTANT : n-2 and not n to consider only the interior intervals 
	// TO BE MODIFIED 
	while(idx < n-2){
		float x_local = xvecGPU[idx + 1]; 
		float sig = interior_sigma(aGPU, bsqrGPU, x_local, gammaGPU, idx + 1, n);
		//printf("%f \n", sig); 
		//float alpha = interior_alpha(sig, &(xvecGPU[idx + 1]), aGPU, idx + 1);
		float ak_local = aGPU[idx + 1]; 
		float ak_minus1_local = aGPU[idx]; 
		float alpha = interior_alpha(sig, x_local, ak_local, ak_minus1_local);
		//yvecGPU[idx + 1] = alpha;
		//yvecGPU[idx + 1] = sig;
		float f = spectral_func(aGPU, bsqrGPU, x_local, gammaGPU, n);
		yvecGPU[idx + 1] = alpha;
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

	//Set gamma
	*gamma = 1;

	// We take the middle of the intervals (initial values from the paper for interior points)
	for (int i=1; i<n-1; i++){
		xvec[i] = (a[i-1] + a[i]) / 2 ;
	}
	
	//Arbitrary filling of the edges values (TO REPLACE BY INITIAL VALUES FROM THE PAPER)
	xvec[0] = a[0] + 5;
	xvec[n-1] = a[n-2] - 5; 


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
	//spectral_func_kernel <<<1024, 512>>> (aGPU, bsqrGPU, yvecGPU, xvecGPU, gammaGPU, n);

	// Transfer spectral function results on CPU to print it
	//cudaMemcpy(yvec, yvecGPU, n*sizeof(float), cudaMemcpyDeviceToHost);
	//print_vector(yvec, n);

	//Compute sigma_interior function on GPU
	//sigma_interior_kernel <<<1024, 512>>> (aGPU, bsqrGPU, yvecGPU, xvecGPU, gammaGPU, n);

	test_all_kernel <<<1024, 512>>> (aGPU, bsqrGPU, yvecGPU, xvecGPU, gammaGPU, n);

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


#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
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



// Compute the squre of a vector
void vector_square(float *b, float *bsqr, int n){

	for (int i=0; i<n-1; i++){
		bsqr[i] = b[i] * b[i];
	}
}


// Compute the squared norm of a vector
// using squared coordinates
float vector_squared_norm(float *bsqr, int n){
	float bnorm=0;
	for (int i=0; i<n-1; i++){
		bnorm += bsqr[i];
	}
	return bnorm;
}



// Compute the initial value for the first eigenvalue
float x0_edge_0(float a_0, float bnorm, float gamma){
	float term1 = (gamma-a_0)/2;
	if (gamma > a_0){
		return a_0 + term1 +sqrtf(term1*term1 + bnorm);
	}
	else{
		return a_0 + bnorm / (-term1 +sqrtf(term1*term1 + bnorm));
	}
}

// Compute the initial value for the last eigenvalue
float x0_edge_nminus2(float a_nminus2, float bnorm, float gamma){
	float term1 = (gamma-a_nminus2)/2;
	if (gamma > a_nminus2){
		return a_nminus2 - term1 -sqrtf(term1*term1 + bnorm);
	}
	else{
		return (float)a_nminus2 - (float)bnorm / (-(float)term1 +(float)sqrtf((float)term1*(float)term1 + (float)bnorm));
	}
}

// Compute f (the spectral function) at a given point x
float spectral_func(float *a,
		    float *bsqr,
		    float x,
		    float gamma,
	    	    int n) {

	float sum = 0;

	for (int i=0; i<n-1; i++){
		sum += bsqr[i] / (a[i] - x);
	}

	return x - gamma + sum;
}


// Compute f' (the prime derivative of the spectral function) at a given point x
float spectral_func_prime(float *a,
			  float *bsqr,
			  float x,
			  int n) {

	float sum = 0;

	for (int i=0; i<n-1; i++){

		int ai_local = a[i];
		sum += bsqr[i] / ((ai_local - x) * (ai_local - x));
	}

	return 1 + sum;
}


// Compute the interior versions of sigma
float interior_sigma(float *a,
		     float *bsqr,
		     float x,
		     float gamma,
		     int k,
		     int n) {

	float sum = 0;

	float ak_local = a[k];
	float ak_minus1_local = a[k-1];

	for (int i=0; i<n-1; i++) {

		//Use the registers
		float ai_local = a[i];

		float num = bsqr[i] * (ai_local - ak_minus1_local) * (ai_local - ak_local);

		float deno = (ai_local - x) * (ai_local - x)
		        * (ai_local - x);

		sum +=  num / deno;
	}

	float term1 = 3 * x - gamma - ak_local - ak_minus1_local;

	return term1 + sum;
}


// Compute the interior version of alpha
float interior_alpha(float sigma, float x, float ak, float ak_minus1){

	return sigma / ((ak_minus1 - x) * (x - ak));
}

// Compute the exterior version of alpha
float exterior_alpha(float x, float *a, int n, int k, float *bsqr){

	float sum = 0;

	if (k==0){
		float a0 = a[0];
		for (int i=1; i<n-1; i++) {
			sum += (bsqr[i]*(a0-a[i]))/((x-a[i])*(x-a[i])*(x-a[i]));
		}
		return -(1+sum)/(x-a0);
	}

	else{
		float a_nminus2 = a[n-2];
		for (int i=0; i<n-2; i++) {
			sum += (bsqr[i]*(a_nminus2-a[i]))/((x-a[i])*(x-a[i])*(x-a[i]));
		}
		return -(1+sum)/(x-a_nminus2);
	}
}


// Compute the interior version of beta
float interior_beta(float fprime, float f, float x, float ak, float ak_minus1){

	float fac = (1 / (ak_minus1 - x) + 1 / (ak - x));
	return fprime - fac * f;

}

// Compute the exterior version of beta
float exterior_beta(float fprime, float f, int k, float x, float a0, float a_nminus2){
	if (k==0){
		return fprime + f/(x-a0);
	}

	else{
		return fprime + f/(x-a_nminus2);
	}
}


// Computation of the update (delta) for the interior intervals
// also valid for exterior intervals
float interior_delta(float f, float alpha, float beta){

	float term1 = 2 * f / beta;
	float term2 = 2 * alpha / beta;
	float deno = 1 + sqrtf(1 + term1 * term2);
	return term1 / deno;
}


//function to find the zero within the interval (a[k], a[k-1])
float interior_zero_finder(float *a,
			   float *bsqr,
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
		float sig = interior_sigma(a, bsqr, x, gamma, k, n);
		float ak_local = a[k];
		float ak_minus1_local = a[k - 1];
		// Computation of alpha(x), see definition (7) of the article in page 8 (13 in the pdf)
		float alpha = interior_alpha(sig, x, ak_local, ak_minus1_local);
		// Computation of spectral_func(x)
		f = spectral_func(a, bsqr, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(a, bsqr, x, n);
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

float exterior_zero_finder(float *a,
			   float *bsqr,
			   float gamma,
			   float x,
				 int k,
			   int n,
			   int maxit,
			   float epsilon){

	int i = 0;
	float a0 = a[0];
	float a_nminus2 = a[n-2];
	// To guarantee entry in the loop
	float f = 2 * sqrtf(epsilon);
	while ((i < maxit) && (f*f > epsilon)){

		// Computation of alpha(x)
		float alpha = exterior_alpha(x, a, n, k, bsqr);
		// Computation of spectral_func(x)
		f = spectral_func(a, bsqr, x, gamma, n);
		// Computation of spectral_func_prime(x)
		float fprime = spectral_func_prime(a, bsqr, x, n);
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

int main (void) {

	/****************** Declaration ******************/
	// Declare vectors or floats
	float bnorm;
	float *a, *b, *bsqr, *x0_vec, *xstar_vec, *c;


	// Gamma
	float gamma = 1;


	// Size of arrow matrix chosen by the user
	int n = 100;
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
	bsqr = (float*)malloc((n-1)*sizeof(float));
	c = (float*)malloc(n*n*sizeof(float));
	x0_vec = (float*)malloc(n*sizeof(float));
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

	/****************** Initialisation ******************/

	// Interior values:
	// We take the middle of the intervals as initial value
	//(as advised in the paper at the beginning of  page 8 (13 of the pdf)
	for (int i=1; i<n-1; i++){
		x0_vec[i] = (a[i-1] + a[i]) / 2 ;
	}


	// Edge values:
	//Compute square of b first and its squared norm
	vector_square(b, bsqr, n);
	bnorm = vector_squared_norm(bsqr, n);

	x0_vec[0] = x0_edge_0(a[0], bnorm, gamma);
	x0_vec[n-1] = x0_edge_nminus2(a[n-2], bnorm, gamma);


	// Fill c with arrow matrix generated from a and b
	//generate_arrow(a, b, c, gamma, n);

	// Print c (not very necessary actually)
	//printf("The arrow matrix : \n");
	//print_matrix(c, n);


	/****************** Root finding ******************/


	// Find first exterior root
	xstar_vec[0] = exterior_zero_finder(a, bsqr, gamma, x0_vec[0], 0, n, maxit, epsilon);

	// Find all n-2 interior roots
	for (int k=1; k<n-1; k++) {
		xstar_vec[k] = interior_zero_finder(a, bsqr, gamma, x0_vec[k], k, n, maxit, epsilon);
	}


	// Find last exterior root
	xstar_vec[n-1] = exterior_zero_finder(a, bsqr, gamma, x0_vec[n-1], n-1, n, maxit, epsilon);


	// End timer
	Tim.add();


	/****************** Displaying ******************/
	// Print the first zeros
	// Number of roots to display
	int m = 10;
	printf("\n********************* RESULTS ********************** \n");
	printf("The first %i greater resulting roots (eigen values) are : \n", m);
	print_vector(xstar_vec, m, n);

	// Print how long it took
	printf("CPU timer for root finding : %f s\n\n",
		(float)Tim.getsum());
	printf("\n");


	/****************** Freeing memory ******************/
	// Free memory on CPU
	free(a);
	free(b);
	free(bsqr);
	free(c);
	free(x0_vec);
	free(xstar_vec);

}

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
// For simplicity's sake, a contains gamma : [a_1,...a_(n-1), gamma]
void generate_arrow(float *a, float *b, float *c, int n) {
	
	int j = 0; 

	// Fill the arrow
	for (int i=0; i<n; i ++){
		
		// Iterate over the diagonal of c
		c[n*i + j] = a[i];
		j ++; 
		
		if (i<n-1) {

		// Iterate over the last column of c
		c[n - 1 + i*n] = b[i];
		
		// Iterate over the last row of c
		c[n * (n-1) + i] = b[i];

		}
	}
}



int main (void) {

	// Reference vectors
	float *a, *b, *c;

	// Size of arrow matrix
	int n = 10;
	
	// Memory allocation
	a = (float*)malloc(n*sizeof(int));
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
	
	// Free memory
	free(a);
	free(b);
	free(c);
	
}


/**********************************************
 *Author: Gurpal Singh
 *
 *Purpose: Computes exact integral, and uses
           Trapezoidal and Simpsons rule
 *
 *Date: April 11, 2017
 *
 *To compile: mpicc -o singh_hw5.exe -O3 singh_hw5.c -lm
 * 
 *To Run: mpirun -np 4 ./singh_hw5.exe 
 * ******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


//Function for what we are integrating
double f(const double x) {
	return 5.0f + 3*(sin(x));
}

//Trapezoidal Rule for Integration
double integrate(const double a , const double b, const unsigned int n){
	
	int i;
	double approx , xi, h;

	h = (b - a) / n;

	approx = (f(a) + f(b)) * 0.5f;

	for(i = 1; i < n; i++){
		xi = a + i * h;
		approx += f(xi);
	}

	return h * approx;

}

//Simpson's Rule for Integration
double simpsons_rule(const double a, const double b, const unsigned int n){

	int i;
	double h, top, bottom, ans;

	h = (b - a)/n;
	
	double y[n], x[n];

	for(i=0; i<=n; i++){
		x[i] = a+i*h;
		y[i] = f(x[i]);
	}

	top = 0;
	bottom = 0;

	for (i=1; i<n; i++){
		if(i%2==1){
			top =top+y[i];
		}
		else{
			bottom = bottom+y[i];
		}

	}

	ans = (h/3) * (y[0] + y[n] + 4*top + 2*bottom);
	return ans;
}


int main(){
	
	//MPI Code
	int nProcs;
	int myRank;

	//Initialize the MPI execution environment	
	MPI_Init(NULL, NULL);

	//Determining the size of the group
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	//Determining the rank of the calling process
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	//Blocks until all processes have reached this routine
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Defining bounds and number of intervals
	double lower_limit, upper_limit;
	unsigned int nIntervals;

	//Scanning for inputs at master processor
	if(myRank == 0){
	
		printf("Enter your lower bound:\n ");
       		scanf("%lf", &lower_limit);
        	printf("Enter your upper bound:\n ");
        	scanf ("%lf", &upper_limit);
        	printf("Enter the number of intervals:\n ");
        	scanf ("%u", &nIntervals);
	
		if(nIntervals < 0) MPI_Abort(MPI_COMM_WORLD, 666);
	}

	//Broadcasting information to all other processes
	MPI_Bcast(&nIntervals, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lower_limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&upper_limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	
	double my_lower_limit = lower_limit + ((upper_limit - lower_limit) / (double) nProcs) * myRank;
	double my_upper_limit = my_lower_limit + (upper_limit - lower_limit) / (double) nProcs;

	double nInterval = nIntervals / nProcs;

	double my_integral = integrate(my_lower_limit, my_upper_limit, nInterval);
	double integral;
	
	double simpson_integral = simpsons_rule(my_lower_limit, my_upper_limit, nInterval);
	double simpson;
	

	//Reducing values on all processes to a single value
	MPI_Reduce(&simpson_integral, &simpson, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&my_integral, &integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//Printing to the Screen
	if(myRank == 0){

		double exactIntegral =(5.0f * upper_limit - 3.0f * cos(upper_limit)) - (5.0f * lower_limit - 3.0f * cos(lower_limit));
		printf("\n");
		printf("integrating the function: 5 + 3sin(x)\n");
		printf("Lower_limit = %f\n", lower_limit);
		printf("Upper_limit = %f\n", upper_limit);
		printf("Numer of intervals = %u\n", nIntervals);
		printf("Exact Integral Solution = %f\n", exactIntegral);
		printf("Trapezoidal Method Solution = %f\n", integral);
		printf("Simpsons Method Solution = %f\n", simpson);

		
    	//Writing The results to a file 
    	FILE *fptr = fopen("Integration_Results.txt", "a+");
    	
	if (fptr == NULL) {
        	printf("Error!");
        	exit(1);
    	}

    	fprintf(fptr, "\n");
    	fprintf(fptr, "Lower Limit:  %f\n", lower_limit);
	fprintf(fptr, "Upper Limit:  %f\n", upper_limit);
    	fprintf(fptr, "Number of Intervals: %u\n", nIntervals);

	fprintf(fptr, "Exact Integral Solution: %f\n", exactIntegral);
	fprintf(fptr, "Trapezoidal Method Solution: %f\n", integral);
	fprintf(fptr, "Simpson's Rule Solution: %f\n", simpson);
    	fclose(fptr);
	}	

	//Blocks until all processes have reached this routine
	MPI_Barrier(MPI_COMM_WORLD);

	//Terminates MPI Environment
	//No More MPI calls after this
	MPI_Finalize();		

	return 0;
}

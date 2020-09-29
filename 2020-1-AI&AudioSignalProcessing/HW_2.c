
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM			10
#define EPOCH		1
#define NUM_TRAINING_SET	1000
#define NUM_OUTPUT	10
#define NUM_HIDDEN	1
#define LR			0.1




FILE* fin;
FILE* fo;
FILE* fo_cost;


float weight_init() {
	return ((float)rand() / (float)RAND_MAX - (float)0.5);
}


void main() {
	float input[DIM] = { 0,0, }, target;
	float weight[DIM] = { 0, 0, };
	//float weight[DIM][NUM_HIDDEN]
	float error = 0;
	float cost[NUM_TRAINING_SET] = { 0,0, };
	


	
	fopen_s(&fo, "output_weight_10ep_0.01LR.dat", "wt");
	fopen_s(&fo_cost,"output_cost_10ep_0.01LR.dat","wt");

	/*weight initialization*/
	for (int i = 0; i < DIM; i++) {
		for (int j = 0; j < NUM_HIDDEN; j++) {
			weight[i] = weight_init();
			//weight[i][j] = weight_init(); hidden node가 2개 이상
			//printf("%f \n", weight[i]);

		}
	}

	/*training*/
	for (int ep = 0; ep < EPOCH; ep++) {//1 epoch
		/*training data load */
		fopen_s(&fin, "trainingDB.dat", "rt");

		for (int x = 0; x < NUM_TRAINING_SET; x++) { // 1000/1=1000
			float output = 0;

			//X_train
			for (int k = 0; k < DIM; k++) {
				fscanf_s(fin, "%f", input + k);
			}
			
			//y_train
			fscanf_s(fin, "%f", &target);

			/* forward */
			for (int j = 0; j < DIM; j++) {
				output += input[j] * weight[j];
			}

			/* backprop */ 

			error = output - target ;
			//printf("output = %f   target = %f   error = %f \n", output, target, error);
			cost[x] = (float)(pow(error, 2))/2;
			for (int i = 0; i < DIM; i++) {
				weight[i] -= (float)(input[i]*LR *(error));
			}
			if (x == 499 || x == 999) {
				printf("%d cost: %10.7f \n", x +1, cost[x]);
			}

			/* weight and cost write */
			
			fprintf(fo_cost, "%10.7f", cost[x]);
			fprintf(fo_cost, "\n");
			for (int i = 0; i < DIM; i++) {
				fprintf(fo, "%10.7f", weight[i]);
			}
			fprintf(fo, "\n");

		}

	}
}


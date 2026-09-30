#include"matrix.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAGIC_NUMBER 0x1404A715

//Reads the first lines of the input file, checks if everything fits and fills the network struct with the correct data. !! The caller needs to free(net) in case of failure
int LoadNetworkHeader(network_t *net, FILE *fp){

    //Here the file is already open and it's existance has already been checked
    int magic_number, i;

    if(fread(&magic_number, sizeof(int), 1, fp) != 1){
        printf("Missing data from file, could not read correctly. \n");
        return 0;
    }
    if(magic_number != MAGIC_NUMBER){
        printf("Invalid file. \n");
        return 0;
    }

    //number of layers in the network 
    if(fread(&net->layer_quantity, sizeof(int), 1, fp) != 1){
        printf("Missing data from file, could not read correctly. \n");
        return 0;
    }

    net->layers = _mm_malloc(sizeof(layer_t) * net->layer_quantity, 32);
    if(net->layers == NULL){
        printf("Memory allocation error. \n");
        return 0;
    }
    for(i = 0; i < net->layer_quantity; i++){
        if(fread(&net->layers[i].input_size, sizeof(int), 1, fp) != 1 || fread(&net->layers[i].output_size, sizeof(int), 1, fp) != 1){
            printf("Missing data from file, could not read correctly. \n");
            _mm_free(net->layers);
            net->layers = NULL;
            return 0;
        }
        
    }
    return 1;
    //the file will be closed only after the LoadNetworkWeights function
}

//Deletes the network struct in case of failure 
void FreeNetPartial(network_t *net, int stop){
    int i;
    for(i = 0; i < stop; i++){
        DeleteMatrix(net->layers[i].weights);
        DeleteMatrix(net->layers[i].biases);
    }
}
//Used after "loadNetworkHeader", it fills the weights and biases matrices with the input data -- !! THE CALLER NEEDS TO CLOSE THE FILE
int LoadNetworkWeights(network_t *net, FILE *fp){
    int i, size;

    net->max_layer_size = 0;

    for(i = 0; i < net->layer_quantity; i++){

        if(net->layers[i].output_size > net->max_layer_size){
            net->max_layer_size = net->layers[i].output_size;
        }

        //Weight
        net->layers[i].weights = CreateMatrix(net->layers[i].input_size, net->layers[i].output_size);

        //Bias
        net->layers[i].biases = CreateMatrix(1, net->layers[i].output_size);

        //If one allocation fails during the loop, every existing matrix needs to be deleted
        if (net->layers[i].weights == NULL || net->layers[i].biases == NULL){
            FreeNetPartial(net, i);
            _mm_free(net->layers);
            net->layers = NULL;
            return 0;
        }

        //reading the weights matrix
        size = net->layers[i].weights->rows * net->layers[i].weights->cols;
        if(fread(net->layers[i].weights->data, sizeof(float), size, fp) != size){
            printf("Missing data to populate weight matrix of layer %d. \n", i);
            FreeNetPartial(net, i + 1);
            _mm_free(net->layers);
            net->layers = NULL;
            return 0;
        }

        //reading the bias vector
        if(fread(net->layers[i].biases->data, sizeof(float), net->layers[i].output_size, fp) != net->layers[i].output_size){
            printf("Missing data to populate bias vector of layer %d. \n", i);
            FreeNetPartial(net, i + 1);
            _mm_free(net->layers);
            net->layers = NULL;
            return 0;
        }
        
    }

    return 1;
}

//The caller function needs to delete the returned matrix
matrix_t* ForwardPass(network_t* net, matrix_t* input){
    matrix_t *current_input, *current_output, *temp;
    int i;
    
    current_input = CreateMatrix(1, net->max_layer_size);
    current_output = CreateMatrix(1, net->max_layer_size);

    if(current_input == NULL && current_output != NULL){
        printf("Memory allocation error. \n");
        DeleteMatrix(current_output);
        return NULL;
    }else if(current_output == NULL && current_input != NULL){
        printf("Memory allocation error. \n");        
        DeleteMatrix(current_input);
        return NULL;
    }else if(current_input == NULL && current_output == NULL){
        printf("Memory allocation error. \n");
        return NULL;
    }

    //copying the data from input to current input
    current_input->cols = input->cols;
    for(i = 0; i < current_input->cols; i++){
        current_input->data[i] = input->data[i];
    }

    for(i = 0; i < net->layer_quantity; i++){
        //the rows and cols of the output matrix are changed inside the function
        MatMul(current_input, net->layers[i].weights, current_output);
        BiasSum(current_output, net->layers[i].biases);

        //rearranging the pointers for the next multiplication
        temp = current_output;
        current_output = current_input;
        current_input = temp;
    }
    DeleteMatrix(current_output);
    current_input = NULL;
    current_output = NULL;

    //temp always contains the latest output
    return temp;

}

double BenchmarkMatMul(matrix_t *A, matrix_t *B, matrix_t *output, int iterations){
    int i;
    double MatMul_time;
    clock_t start, end;

    start = clock();
    for(i = 0; i < iterations; i++){
        Choose_Matmul_Method(A,B,output);
    }
    end = clock();

    printf("%f\n\n", output->data[0]);

    MatMul_time = ((double)(end - start) / CLOCKS_PER_SEC);
    MatMul_time /= iterations;
    
    return MatMul_time;
}
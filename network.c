#include"matrix.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "quantization.h"
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

        //The ping pong buffers must contain both the inputs and the outputs of every layer
        if(net->layers[i].output_size > net->max_layer_size){
            net->max_layer_size = net->layers[i].output_size;
        }
        if(net->layers[i].input_size > net->max_layer_size){
            net->max_layer_size = net->layers[i].input_size;
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
    matrix_t *current_input, *current_output, *temp = NULL;
    int i;

    if(net == NULL || input == NULL || net->layer_quantity <= 0 || input->rows != 1 || input->cols != net->layers[0].input_size){
        printf("Invalid network or input. \n");
        return NULL;
    }

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
        Choose_Matmul_Method(current_input, net->layers[i].weights, current_output);
        BiasSum(current_output, net->layers[i].biases);
        //Without an activation function the whole network would collapse into a single linear layer
        if(i < net->layer_quantity - 1)
            ReLU(current_output);

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

void FreeQNetwork(q_network_t *net){
    int i;

    if(net == NULL || net->layers == NULL)
        return;

    for(i = 0; i < net->layer_quantity; i++){
        deleteMatQ8(net->layers[i].weights);
        DeleteMatrix(net->layers[i].biases);
    }
    _mm_free(net->layers);
    net->layers = NULL;
    net->layer_quantity = 0;
}

//The weights are constant, so they are quantized only once here and not during every forward pass
int QuantizeNetwork(const network_t *src, q_network_t *dst){
    int i, j;

    if(src == NULL || dst == NULL || src->layers == NULL || src->layer_quantity <= 0){
        printf("Invalid network. \n");
        return -1;
    }

    dst->layers = _mm_malloc(sizeof(q_layer_t) * src->layer_quantity, 32);
    if(dst->layers == NULL){
        printf("Memory allocation error. \n");
        return -1;
    }
    dst->layer_quantity = src->layer_quantity;
    dst->max_layer_size = src->max_layer_size;

    //Every pointer starts at NULL so that FreeQNetwork can always be used in case of failure
    for(i = 0; i < dst->layer_quantity; i++){
        dst->layers[i].weights = NULL;
        dst->layers[i].biases = NULL;
    }

    for(i = 0; i < src->layer_quantity; i++){
        const layer_t *l = &src->layers[i];

        dst->layers[i].input_size = l->input_size;
        dst->layers[i].output_size = l->output_size;
        dst->layers[i].weights = createMatrix8(l->input_size, l->output_size);
        dst->layers[i].biases = CreateMatrix(1, l->output_size);

        if(dst->layers[i].weights == NULL || dst->layers[i].biases == NULL){
            FreeQNetwork(dst);
            return -1;
        }

        if(quantize(l->weights, dst->layers[i].weights, 127.0f) != 0){
            printf("Could not quantize the weights of layer %d. \n", i);
            FreeQNetwork(dst);
            return -1;
        }

        for(j = 0; j < l->output_size; j++)
            dst->layers[i].biases->data[j] = l->biases->data[j];
    }

    return 0;
}

//The caller function needs to delete the returned matrix
matrix_t* QForwardPass(q_network_t* net, matrix_t* input){
    matrix_t *activation = NULL;
    Q_matrix8_t *activation_q = NULL;
    Q_matrix32_t *acc = NULL;
    int i, rows, buffer_size, last;

    if(net == NULL || input == NULL || net->layers == NULL || net->layer_quantity <= 0 ||
       input->rows <= 0 || input->cols != net->layers[0].input_size){
        printf("Invalid network or input. \n");
        return NULL;
    }

    rows = input->rows;
    buffer_size = net->max_layer_size;

    //Buffers allocated only once: the float activation is overwritten in place by every layer
    activation = CreateMatrix(rows, buffer_size);
    activation_q = createMatrix8(rows, buffer_size);
    acc = createMatrix32(rows, buffer_size);
    if(activation == NULL || activation_q == NULL || acc == NULL){
        DeleteMatrix(activation);
        deleteMatQ8(activation_q);
        deleteMatQ32(acc);
        return NULL;
    }

    activation->rows = rows;
    activation->cols = input->cols;
    for(i = 0; i < rows * input->cols; i++)
        activation->data[i] = input->data[i];

    for(i = 0; i < net->layer_quantity; i++){
        last = (i == net->layer_quantity - 1);

        //1. dynamic quantization of the activation (a new scale for every input)
        //2. int8 x int8 matmul accumulated in int32, scale = s_activation * s_weights
        //3. back to float adding the bias, ReLU on every layer except the last one
        if(quantize(activation, activation_q, 127.0f) != 0 ||
           qMatmulAcc32(activation_q, net->layers[i].weights, acc, NULL) != 0 ||
           dequantizeAcc32(acc, net->layers[i].biases, activation, !last) != 0){
            printf("Error during the quantized forward pass at layer %d. \n", i);
            DeleteMatrix(activation);
            activation = NULL;
            break;
        }
    }

    deleteMatQ8(activation_q);
    deleteMatQ32(acc);

    return activation;
}

//Wall-clock time (omp_get_wtime, clock() would sum the CPU time of every thread) of one quantized matmul (int8 -> int32 accumulator)
double BenchmarkQMatMul(const Q_matrix8_t *A, const Q_matrix8_t *B, Q_matrix32_t *acc, int iterations){
    int i;
    double start;

    if(iterations <= 0 || qMatmulAcc32(A, B, acc, NULL) != 0)   //warm-up + validation
        return -1.0;

    start = omp_get_wtime();
    for(i = 0; i < iterations; i++)
        qMatmulAcc32(A, B, acc, NULL);

    return (omp_get_wtime() - start) / iterations;
}

//Wall-clock time of one forward pass. quantized = 0 -> ForwardPass (float, input 1 x n), quantized = 1 -> QForwardPass
double BenchmarkForward(network_t *net, q_network_t *qnet, matrix_t *input, int iterations, int quantized){
    int i;
    double start;
    matrix_t *out;

    if(iterations <= 0 || (quantized ? qnet == NULL : net == NULL))
        return -1.0;

    start = omp_get_wtime();
    for(i = 0; i < iterations; i++){
        out = quantized ? QForwardPass(qnet, input) : ForwardPass(net, input);
        if(out == NULL)
            return -1.0;
        DeleteMatrix(out);
    }

    return (omp_get_wtime() - start) / iterations;
}

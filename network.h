#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"
#include <stdio.h>

typedef struct{
    matrix_t *weights;
    matrix_t *biases;
    int input_size;
    int output_size;
}layer_t;


typedef struct{
    layer_t *layers;
    int layer_quantity;
    int max_layer_size;
}network_t;

int LoadNetworkHeader(network_t *net, FILE *fp);
int LoadNetworkWeights(network_t *net, FILE *fp);
void FreeNetPartial(network_t* net, int stop);
matrix_t* ForwardPass(network_t* net, matrix_t* input);
double BenchmarkMatMul(matrix_t *A, matrix_t *B, matrix_t *out, int iterations);
#endif
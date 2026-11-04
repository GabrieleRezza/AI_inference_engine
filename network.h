#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"
#include "quantization.h"
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
    int max_layer_size;     //max between every input_size and output_size, used to size the ping pong buffers
}network_t;

//Quantized layer: int8 weights (per-tensor scale), float biases added after the dequantization of the accumulator
typedef struct{
    Q_matrix8_t *weights;
    matrix_t *biases;
    int input_size;
    int output_size;
}q_layer_t;

typedef struct{
    q_layer_t *layers;
    int layer_quantity;
    int max_layer_size;
}q_network_t;

int LoadNetworkHeader(network_t *net, FILE *fp);
int LoadNetworkWeights(network_t *net, FILE *fp);
void FreeNetPartial(network_t* net, int stop);
//Forward pass of the float network, ReLU is applied after every layer except the last one
matrix_t* ForwardPass(network_t* net, matrix_t* input);
double BenchmarkMatMul(matrix_t *A, matrix_t *B, matrix_t *out, int iterations);

//Builds a quantized copy of a loaded float network (weights quantized once, biases copied). Return -1 if any error occurs
int QuantizeNetwork(const network_t *src, q_network_t *dst);
//Frees every matrix of the quantized network and its layers array
void FreeQNetwork(q_network_t *net);
/*Forward pass with dynamic quantization: for every layer the float activation is quantized (int8), multiplied
  with the int8 weights (int32 accumulator) and brought back to float adding bias and ReLU (except on the last layer).
  input is rows x input_size. The caller function needs to delete the returned matrix */
matrix_t* QForwardPass(q_network_t* net, matrix_t* input);
//Average wall-clock seconds per qMatmulAcc32 (A x B -> acc). Return -1 if any error occurs
double BenchmarkQMatMul(const Q_matrix8_t *A, const Q_matrix8_t *B, Q_matrix32_t *acc, int iterations);
//Average wall-clock seconds per forward pass: quantized = 0 float ForwardPass (input 1 x n), quantized = 1 QForwardPass. Return -1 on error
double BenchmarkForward(network_t *net, q_network_t *qnet, matrix_t *input, int iterations, int quantized);
#endif

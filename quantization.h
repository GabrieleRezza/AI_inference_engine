#ifndef MATRIXQ_H
#define MATRIXQ_H
#include <stdint.h>
#include "matrix.h"

typedef struct{
    int *data_32;
    int rows;
    int cols;
    float scale_factor;
}Q_matrix32_t;

typedef struct{
    int8_t *data_8;
    int rows;
    int cols;
    float scale_factor;
}Q_matrix8_t;
//Create matrix with int32 data and a scale_factor attribute
Q_matrix32_t *createMatrix32(int row, int col);
//Create matrix with int8 data and a scale_factor attribute
Q_matrix8_t *createMatrix8(int row, int col);
//Delete a type Q_matrix32_t matrix 
void deleteMatQ32(Q_matrix32_t *mat);
//Delete a type Q_matrix8_t matrix 
void deleteMatQ8(Q_matrix8_t *mat);
/*Matmul funtion using quantization, int8 data type multiplied into an int16 register and accumulated in an int32.
 The final result is ri-quantized again before the end of the function. 
 NOTE: the elements are shifted by 127 / 63 directly in the matmul during SIMD*/
int qMatmulSIMD(Q_matrix8_t *A, Q_matrix8_t *B, Q_matrix32_t *acc, Q_matrix8_t *output);
//Quantize a float matrix into an int8 matrix, return -1 if any error occurs
int quantize(matrix_t *origin, Q_matrix8_t *output, float max_value);
//Rebuilds the original float matrix from an int8 matrix thanks to the scale factor. return -1 if any error occurs 
int dequantize(Q_matrix8_t *mat8, matrix_t *matf);
//Requantize the int32 matrix into a int8 matrix. If error return -1
int requantize(Q_matrix32_t* input, Q_matrix8_t* output, float max_value);
//Requantize from int32 to int8 using SIMD and threading
int requantizeSIMD(const Q_matrix32_t *input, Q_matrix8_t *output, float max_value);



#endif
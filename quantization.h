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

/*Scale convention used everywhere: q = round(x * scale_factor)  -->  x = q / scale_factor
  NOTE: every function that writes into an output matrix sets its rows/cols; the caller must guarantee
  that the output buffer is big enough (rows*cols elements). Quantized values are always kept in [-127, 127]. */

//Create matrix with int32 data and a scale_factor attribute
Q_matrix32_t *createMatrix32(int row, int col);
//Create matrix with int8 data and a scale_factor attribute
Q_matrix8_t *createMatrix8(int row, int col);
//Delete a type Q_matrix32_t matrix
void deleteMatQ32(Q_matrix32_t *mat);
//Delete a type Q_matrix8_t matrix
void deleteMatQ8(Q_matrix8_t *mat);
//Reset an int8 matrix to 0
void CleanInt8Mat(Q_matrix8_t *mat);
//Reset an int32 matrix to 0
void CleanInt32Mat(Q_matrix32_t *mat);
/*Int8 x int8 matmul (SIMD + threading) accumulated in int32: acc = A*B (+ bias).
  acc->scale_factor is set to A->scale_factor * B->scale_factor, so the bias (optional, NULL = no bias, 1 x B->cols)
  must already be quantized with that same scale: bias_q = round(bias_f * sA * sB). Return -1 if any error occurs */
int qMatmulAcc32(const Q_matrix8_t *A, const Q_matrix8_t *B, Q_matrix32_t *acc, const Q_matrix32_t *bias);
/*Matmul funtion using quantization: qMatmulAcc32 followed by requantizeSIMD, the final int8 result is in output.
  bias is optional (NULL) and follows the same rules of qMatmulAcc32. Return -1 if any error occurs */
int qMatmulSIMD(const Q_matrix8_t *A, const Q_matrix8_t *B, Q_matrix32_t *acc, Q_matrix8_t *output, const Q_matrix32_t *bias);
//Quantize a float matrix into an int8 matrix (SIMD + threading) with values in [-qmax, qmax], qmax <= 127. Return -1 if any error occurs
int quantize(const matrix_t *origin, Q_matrix8_t *output, float max_value);
//Rebuilds the original float matrix from an int8 matrix thanks to the scale factor. return -1 if any error occurs
int dequantize(Q_matrix8_t *mat8, matrix_t *matf);
//Rebuild the original float matrix from an int8 matrix using SIMD and threading
int dequantizeSIMD(const Q_matrix8_t *input, matrix_t *output);
/*Bring an int32 accumulator back to float in a single pass: output = acc / acc->scale_factor + bias, with optional ReLU.
  bias is a float 1 x acc->cols matrix (NULL = no bias). Return -1 if any error occurs */
int dequantizeAcc32(const Q_matrix32_t *acc, const matrix_t *bias, matrix_t *output, int relu);
//Requantize the int32 matrix into a int8 matrix. If error return -1
int requantize(Q_matrix32_t* input, Q_matrix8_t* output, float max_value);
//Requantize from int32 to int8 using SIMD and threading
int requantizeSIMD(const Q_matrix32_t *input, Q_matrix8_t *output, float max_value);



#endif

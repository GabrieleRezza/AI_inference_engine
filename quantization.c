#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include "matrix.h"
#include "quantization.h"

Q_matrix32_t *createMatrix32(int rows, int cols){
    Q_matrix32_t *mat;
    int32_t* data_p;

    mat = _mm_malloc(sizeof(Q_matrix32_t), 32);
    if (mat == NULL){
        printf("Memory allocation error. \n");
        return NULL;
    }

    data_p = (int32_t *)_mm_malloc(sizeof(int32_t) * rows * cols, 32);
    if (data_p == NULL){
        printf("Memory allocation error. \n");
        _mm_free(mat);
        return NULL;
    }
    
    mat->cols = cols;
    mat->rows = rows;
    mat->data_32 = data_p;
    mat->scale_factor = 1;

    return mat;
}

Q_matrix8_t *createMatrix8(int rows, int cols){
    Q_matrix8_t *mat;
    int8_t* data_p;

    mat = _mm_malloc(sizeof(Q_matrix8_t), 32);
    if (mat == NULL){
        printf("Memory allocation error. \n");
        return NULL;
    }

    data_p = (int8_t *)_mm_malloc(sizeof(int8_t) * rows * cols, 32);
    if (data_p == NULL){
        printf("Memory allocation error. \n");
        _mm_free(mat);
        return NULL;
    }
    
    mat->cols = cols;
    mat->rows = rows;
    mat->data_8 = data_p;
    mat->scale_factor = 1;

    return mat;
}

void deleteMatQ32(Q_matrix32_t *mat){
    if(mat == NULL){
        return;
    }

    _mm_free(mat->data_32);
    _mm_free(mat);
}

void deleteMatQ8(Q_matrix8_t *mat){
    if(mat == NULL){
        return;
    }

    _mm_free(mat->data_8);
    _mm_free(mat);
}
//max value == 127 or
int quantize(matrix_t *input, Q_matrix8_t *mat8, float max_value){
    float abs_value, max_abs = 0.0f, temp;
    int i, mat_dim;
    if(input == NULL || mat8 == NULL || input->rows == 0 || input->cols == 0){
        printf("Empty matrix \n");
        return -1;
    }
    
    mat_dim = input->cols * input->rows;
    //searching for max value of the matrix
    for(i = 0; i < mat_dim; i++){
        abs_value = fabsf(input->data[i]);

        if (abs_value > max_abs)
            max_abs = abs_value;
    }

    //if max_abs == 0 I force into 1
    if (max_abs == 0.0f)
        mat8->scale_factor = 1.0f;
    else{
        mat8->scale_factor = max_value / max_abs;
        if (!isfinite(mat8->scale_factor)) {   // inf (max_abs) o NaN
            printf("Invalid scale factor \n");
            return -1;
        }
    }
    //copying the matrix into the int8 version
    for(i = 0; i < mat_dim; i++){
        temp =  roundf(input->data[i] * mat8->scale_factor);
        //technically the quantized values should not overflow since even the scale factor is set on max_abs
        //we still clamp for good measure
        if (temp > max_value)
            temp = max_value;
        else if(temp < -max_value )
            temp = -max_value;

        mat8->data_8[i] = (int8_t)temp;
    }
    return 0;
}

int dequantize(Q_matrix8_t* input, matrix_t* output){
    int i, mat_dim;
    float dequantize_factor;

    if( input == NULL || output == NULL || input->rows == 0 || input->cols == 0){
        printf("Invalid input \n");
        return -1;
    }
    mat_dim = input->cols * input->rows;
    if (input->scale_factor == 0.0f)
        dequantize_factor = 1.0f;

    dequantize_factor = 1.0f / input->scale_factor;
    for(i = 0; i < mat_dim; i++){
        output->data[i] = (float) input->data_8[i] * dequantize_factor;
    }
    return 0;
}


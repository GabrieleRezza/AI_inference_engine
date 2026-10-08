#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
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

void quantize(matrix_t *origin, Q_matrix8_t *mat8){

}


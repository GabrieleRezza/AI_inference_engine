#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "matrix.h"
#define BLOCK_SIZE 128

int min(int A, int B){
    if(A <= B)
        return A;
    else{
        return B;
    }
}

matrix_t* CreateMatrix(int rows, int cols){
    matrix_t *mat;
    float* data_p;

    mat = _mm_malloc(sizeof(matrix_t), 32);
    if (mat == NULL){
        printf("Memory allocation error. \n");
        return NULL;
    }

    data_p = _mm_malloc(sizeof(float) * rows * cols, 32);
    if (data_p == NULL){
        printf("Memory allocation error. \n");
        _mm_free(mat);
        return NULL;
    }
    
    mat->cols = cols;
    mat->rows = rows;
    mat->data = data_p;

    return mat;
}

void DeleteMatrix(matrix_t* mat){
    if(mat == NULL){
        return;
    }

    _mm_free(mat->data);
    _mm_free(mat);
}

void InitMatrixRandom(matrix_t *A){
    int i;

    if(A == NULL){
        printf("Matrix pointer NULL. \n");
        return;
    }

    srand(time(NULL));

    for(i = 0; i < A->rows * A->cols; i++)
        A->data[i] = ((float)rand() / RAND_MAX) * 10.0f;
}


void CleanMatrix(matrix_t* mat){
    int i;

    if(mat == NULL){
        printf("NULL matrix. \n");
        return;
    }
    
    for(i = 0; i < mat->rows*mat->cols; i++)
        mat->data[i] = 0;
}

void PrintMatrix(matrix_t* mat){
    int i,j;

    if(mat == NULL){
        printf("Matrix pointer is NULL. \n");
        return;
    }

    for(i = 0; i < mat->rows; i++){
       for(j = 0; j < mat->cols; j++){
            printf(" [%f] ", mat->data[i * mat->cols + j]);
       }
       printf("\n");
    }
}


float GetData(matrix_t* mat, int row, int col){

    if(row >= mat->rows || col >= mat->cols || row < 0 || col < 0){
        printf("Invalid search position. \n");
        return 0;
    }

    return mat->data[row * mat->cols + col];
}

void MatMul(matrix_t* A, matrix_t* B, matrix_t* output){
    int i, j, k, block_i, block_k, block_j, min_i, min_j, min_k;
    
    //first we need to check if the multiplication is actually possible, even if it was probably checked before
    if(A->cols != B->rows){
        printf("Multiplication impossible, wrong dimensions. \n");
        return;
    }
    output->rows = A->rows;
    output->cols = B->cols;
    //First we need to clean the output matrix from the previous results
    CleanMatrix(output);
    for(block_i = 0; block_i < output->rows; block_i += BLOCK_SIZE)
        for(block_k = 0; block_k < B->rows; block_k += BLOCK_SIZE){
            for(block_j = 0; block_j < B->cols; block_j += BLOCK_SIZE){
                min_i = min(block_i + BLOCK_SIZE, output->rows);
                min_j = min(block_j + BLOCK_SIZE, B->cols);
                min_k = min(block_k + BLOCK_SIZE, B->rows);
                for(i = block_i; i < min_i; i++)
                    for(k = block_k; k < min_k; k++)
                        for(j = block_j; j < min_j; j++)
                            output->data[i*output->cols + j] += A->data[i*A->cols + k] * B->data[k*B->cols + j];
                    
            }
        }
}

void MatMul_SIMD_Blocks(matrix_t* A, matrix_t* B, matrix_t* output){
    int i, j, k, block_i, block_k, block_j, min_i, min_j, min_k;
    
    //first we need to check if the multiplication is actually possible, even if it was probably checked before
    if(A->cols != B->rows){
        printf("Multiplication impossible, wrong dimensions. \n");
        return;
    }
    output->rows = A->rows;
    output->cols = B->cols;
    //First we need to clean the output matrix from the previous results
    CleanMatrix(output);

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel private(i,j,k, block_i, block_j, block_k, min_i, min_j, min_k) shared(A, B, output) 
    {
        for(block_i = 0; block_i < output->rows; block_i += BLOCK_SIZE){
            for(block_j = 0; block_j < B->cols; block_j += BLOCK_SIZE){
                for(block_k = 0; block_k < B->rows; block_k += BLOCK_SIZE){
                    min_i = min(block_i + BLOCK_SIZE, output->rows);
                    min_j = min(block_j + BLOCK_SIZE, B->cols);
                    min_k = min(block_k + BLOCK_SIZE, B->rows);

                    //Creating a parallel for loop by collapsing the i and j loops to speed up the multiplication process
                    //I can only collapse two loops because the k loop has a cumulative effect on the output matrix, so it cannot be parallelized
                    #pragma omp for collapse(2)  
                    for(i = block_i; i < min_i; i++){
                        for(j = block_j; j < min_j - (min_j % 8); j += 8){

                            __m256 rc = _mm256_loadu_ps(&output->data[i*output->cols + j]);

                            for(k = block_k; k < min_k; k++){

                                __m256 ra = _mm256_set1_ps(A->data[i*A->cols + k]);
                                    
                                __m256 rb = _mm256_loadu_ps(&B->data[k*B->cols + j]);
                                
                                rc = _mm256_fmadd_ps(ra, rb, rc);
                            }

                            _mm256_storeu_ps(&output->data[i*output->cols + j], rc);

                        }
                    }
                    
                    //Handling the remaining columns that are not a multiple of 8 outside of the parallelized loop
                    #pragma omp for
                    for(i = block_i; i < min_i; i++){
                        if((min_j % 8) != 0){
                                for(j = min_j - (min_j % 8); j < min_j; j++)
                                    for(k = block_k; k < min_k; k++)
                                        output->data[i*output->cols + j] += A->data[i*A->cols + k] * B->data[k*B->cols + j];
                        }
                    }
                }
            }
        }
    }
}

void MatMul_SIMD_Vector(matrix_t* A, matrix_t* B, matrix_t* output){
    int i, j, k;
    
    //first we need to check if the multiplication is actually possible, even if it was probably checked before
    if(A->cols != B->rows){
        printf("Multiplication impossible, wrong dimensions. \n");
        return;
    }
    output->rows = A->rows;
    output->cols = B->cols;
    //First we need to clean the output matrix from the previous results
    CleanMatrix(output);

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel private(i,j,k) shared(A, B, output) 
    {
        #pragma omp for collapse(2)
        for(i = 0; i < output->rows; i++){  
            for(j = 0; j < output->cols - (output->cols % 8); j += 8){

                __m256 rc = _mm256_loadu_ps(&output->data[i*output->cols + j]);

                for(k = 0; k < A->cols; k++){

                    __m256 ra = _mm256_set1_ps(A->data[i*A->cols + k]);
                        
                    __m256 rb = _mm256_loadu_ps(&B->data[k*B->cols + j]);
                    
                    rc = _mm256_fmadd_ps(ra, rb, rc);
                }

                _mm256_storeu_ps(&output->data[i*output->cols + j], rc);

            }
        }
        
        //Handling the remaining columns that are not a multiple of 8 outside of the parallelized loop
        #pragma omp for
        for(i = 0; i < output->rows; i++){
            if((output->cols % 8) != 0){
                    for(j = output->cols - (output->cols % 8); j < output->cols; j++)
                        for(k = 0; k < A->cols; k++)
                            output->data[i*output->cols + j] += A->data[i*A->cols + k] * B->data[k*B->cols + j];
            }
        }
    }
}

void Choose_Matmul_Method(matrix_t* A, matrix_t* B, matrix_t* output){
    if(A->rows == 1){
        MatMul_SIMD_Vector(A, B, output);
    }
    else{
        MatMul_SIMD_Blocks(A, B, output);
    }
}

void BiasSum(matrix_t* output, matrix_t* bias){
    int i;

    if (output == NULL || bias == NULL || output->rows != bias->rows || output->cols != bias->cols){
        printf("Invalid inputs. \n");
        return;
    }

    for(i = 0; i < output->rows * output->cols; i++){
        output->data[i] += bias->data[i];
    }

}


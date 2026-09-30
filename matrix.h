#ifndef MATRIX_H
#define MATRIX_H
//Since allocating a normal matrix is inconvenient and inefficient, a 1D array is used instead, with rows and cols indicated
typedef struct{
    float *data;
    int rows;
    int cols;
}matrix_t;


matrix_t* CreateMatrix(int row, int col);
void DeleteMatrix(matrix_t* A);
void MatMul(matrix_t* A, matrix_t* B, matrix_t* output);
void Choose_Matmul_Method(matrix_t* A, matrix_t* B, matrix_t* output);
void MatMul_SIMD_Blocks(matrix_t* A, matrix_t* B, matrix_t* output);
void MatMul_SIMD_Vector(matrix_t* A, matrix_t* B, matrix_t* output);
void BiasSum(matrix_t* A, matrix_t* bias);
void InitMatrixRandom(matrix_t* A);
void CleanMatrix(matrix_t* A);
float GetData(matrix_t* mat, int row, int col);
int min(int A, int B);
//Auxiliary function for debuggingg
void PrintMatrix(matrix_t* A);


#endif
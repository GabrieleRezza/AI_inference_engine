#include "../matrix.h"
#include "../network.h"
#include <time.h>
#include <stdlib.h>


int main(){
    matrix_t *A, *B, *output;

    A = CreateMatrix(2, 2);
    if(A == NULL)
        return 0;

    B = CreateMatrix(2, 3);
    if(B == NULL){
        DeleteMatrix(A);
        return 0;
    }

    output = CreateMatrix(2, 3);
    if(output == NULL){
        DeleteMatrix(A);
        DeleteMatrix(B);
        return 0;
    }

    A->data[0] = 1;
    A->data[1] = 2;
    A->data[2] = 0.5;
    A->data[3] = -1;
    B->data[0] = 0;
    B->data[1] = 4;
    B->data[2] = 0.5;
    B->data[3] = -2;
    B->data[4] = 1;
    B->data[5] = -1;
    PrintMatrix(A);
    PrintMatrix(B);
    Choose_Matmul_Method(A, B, output);

    PrintMatrix(output);
    DeleteMatrix(A);
    DeleteMatrix(B);
    DeleteMatrix(output);

    return 0;
}
#include "../matrix.h"
#include "../network.h"
#include <time.h>
#include <stdlib.h>


int main(){
    matrix_t *A, *B, *output;
    double average_time;

    A = CreateMatrix(512, 512);
    if(A == NULL)
        return 0;

    B = CreateMatrix(512, 512);
    if(B == NULL){
        DeleteMatrix(A);
        return 0;
    }

    output = CreateMatrix(512, 512);
    if(output == NULL){
        DeleteMatrix(A);
        DeleteMatrix(B);
        return 0;
    }
    InitMatrixRandom(A);
    InitMatrixRandom(B);
    average_time = BenchmarkMatMul(A, B, output, 10);
 
    printf("Tempo medio per operazione: %f\n\n", average_time);

    DeleteMatrix(A);
    DeleteMatrix(B);
    DeleteMatrix(output);
    printf("eliminated\n");
    return 0;
}
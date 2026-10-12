#include "../matrix.h"
#include "../quantization.h"
#include <stdlib.h>
#include <stdio.h>


int main(){
    matrix_t *A; 
    Q_matrix8_t *output;

    A = CreateMatrix(2, 3);
    if(A == NULL)
        return 0;

    output = createMatrix8(2, 3);
    if(output == NULL){
        DeleteMatrix(A);
        return 0;
    }

    A->data[0] = 5.25;
    A->data[1] = 0.13;
    A->data[2] = 2.4;
    A->data[3] = -3.7;
    A->data[4] = -1.7;
    A->data[5] = -0.23;

    PrintMatrix(A);
    
    if (quantize(A, output, 63) == -1){
        printf("error \n\n");
        DeleteMatrix(A);
        deleteMatQ8(output);
        return 0;
    }

    if(dequantize(output, A) == -1){
        printf("error \n\n");
        DeleteMatrix(A);
        deleteMatQ8(output);
        return 0;
    }

    PrintMatrix(A);

    DeleteMatrix(A);
    deleteMatQ8(output);

    return 0;
}
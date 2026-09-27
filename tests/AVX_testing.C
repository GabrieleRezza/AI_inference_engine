#include <immintrin.h>
#include <stdio.h>

void AVX_sum(float* a, float* b, float* c, int n);

//This example function sums the elements of a and b as follows: a[i] + b[i] = c[i]
void AVX_sum(float* a, float* b, float* c, int n){
    int i;

    // At first we use the all register to sum numbers, once we reach the last 0-7 floats we need to be careful
    for(i = 0; i < n - (n % 8); i+= 8){
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        __m256 vres = _mm256_add_ps(va, vb);
        _mm256_store_ps(&c[i], vres);
    }

    for(i = i; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    float c[16];
    int n = 16, i;

    AVX_sum(a,b,c,n);
    for(i = 0; i < n; i++){
        printf("%f  ", c[i]);
    }
    
    return 0;
}
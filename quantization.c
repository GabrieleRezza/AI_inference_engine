#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include "matrix.h"
#include "quantization.h"

#define REQ_PAR_THRESHOLD 65536   /* sotto questa soglia niente parallelismo: da tarare col benchmark */

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




int requantize(Q_matrix32_t* input, Q_matrix8_t* output, float max_value){
    int i, mat_dim;
    int32_t abs_value, max_abs = 0;
    float temp, rescale;

    if(input == NULL || output == NULL || input->rows == 0 || input->cols == 0){
        printf("Empty matrix \n");
        return -1;
    }

    mat_dim = input->cols * input->rows;


    for(i = 0; i < mat_dim; i++){
        //we assume input->data[i] is far smaller than 2^31

        if(input->data_32[i] > 0)
            abs_value = input->data_32[i];
        else{
            abs_value = -input->data_32[i];
        }

        if (abs_value > max_abs)
            max_abs = abs_value;
    }

    //The new scale factor needs to know the previous scale factors to later bring the int back to float
    if(max_abs == 0){
        rescale = 0.0f;
        output->scale_factor = 1.0f;
    }else{
        rescale = max_value / (float)max_abs;

        //scale factor of input is already set to the product of the two previous multiplications
        output->scale_factor = rescale * input->scale_factor;
        if (!isfinite(output->scale_factor)) {   
            printf("Invalid scale factor \n");
            return -1;
        }
    }

    for (i = 0; i < mat_dim; i++){
        temp = roundf((float)input->data_32[i] * rescale);

        if (temp > max_value)
            temp = max_value;
        else if(temp < -max_value)
            temp = -max_value;

        output->data_8[i] = (int8_t)temp;
    }

    return 0;
}


static inline int32_t hmax_epi32_avx2(__m256i v) {
    __m128i m = _mm_max_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(m);
}


int requantizeSIMD(const Q_matrix32_t *input, Q_matrix8_t *output, float max_value) {
    const int mat_dim = input->rows * input->cols;
    const int vec_end = mat_dim & ~7;         //cuts off the last 7 elements if needed
    const int par = (mat_dim >= REQ_PAR_THRESHOLD);
    int32_t max_abs = 0;
    float rescale;

    
    
    if (input == NULL || output == NULL || input->rows == 0 || input->cols == 0) {
        printf("Empty or invalid matrix\n");
        return -1;
    }
    if (!(input->scale_factor > 0.0f)) {      /* intercetta 0, negativi e NaN */
        printf("Invalid input scale factor\n");
        return -1;
    }
    assert(output->rows == input->rows && output->cols == input->cols);



    /* max_abs */
    #pragma omp parallel if(par)
    {
        __m256i v_max_loc = _mm256_setzero_si256();

        #pragma omp for nowait
        for (int i = 0; i < vec_end; i += 8) {
            __m256i v = _mm256_loadu_si256((const __m256i *)&input->data_32[i]);
            v_max_loc = _mm256_max_epi32(v_max_loc, _mm256_abs_epi32(v));
        }

        int32_t t_max = hmax_epi32_avx2(v_max_loc);
        #pragma omp critical
        { if (t_max > max_abs) max_abs = t_max; }
    }

    //cheching the last elements that do not fit in a 256 register ( less than 8 int )
    for (int i = vec_end; i < mat_dim; i++) {
        int32_t v = input->data_32[i];
        int32_t a = (v < 0) ? -v : v;
        if (a > max_abs) max_abs = a;
    }

    /* scaling */
    if (max_abs == 0) {
        rescale = 0.0f;
        output->scale_factor = 1.0f;
    } else {
        rescale = max_value / (float)max_abs;
        output->scale_factor = input->scale_factor * rescale;
        if (!isfinite(output->scale_factor)) {
            printf("Invalid scale factor\n");
            return -1;
        }
    }

    /* scaling, clampimg, round-to-nearest-even (cvtps), pack into int8 */
    const __m256 v_scale = _mm256_set1_ps(rescale);
    const __m256 v_hi    = _mm256_set1_ps(max_value);
    const __m256 v_lo    = _mm256_set1_ps(-max_value);

    #pragma omp parallel for schedule(static) if(par)
    for (int i = 0; i < vec_end; i += 8) {

        __m256 f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)&input->data_32[i]));

        /*This clamping could be eliminated since packs already saturates to 127 max and the eventual errors
        are only caused only on the max value, irrilevant. Despite this, it is kept for generality and for
        clamping values on max_value < 127 */
        f = _mm256_mul_ps(f, v_scale);
        f = _mm256_min_ps(_mm256_max_ps(f, v_lo), v_hi);        //clamping low after clamping high inside

        __m256i r32 = _mm256_cvtps_epi32(f);

        __m128i v16 = _mm_packs_epi32(_mm256_castsi256_si128(r32),
                                      _mm256_extracti128_si256(r32, 1)/*picks the upper half of r32*/);
        __m128i v8  = _mm_packs_epi16(v16, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)&output->data_8[i], v8);
    }

    /* lrintf = same clamping of cvtps */
    for (int i = vec_end; i < mat_dim; i++) {
        float t = (float)input->data_32[i] * rescale;
        if (t > max_value)       
            t = max_value;
        else if (t < -max_value) 
            t = -max_value;
        output->data_8[i] = (int8_t)lrintf(t);
    }

    return 0;
}



int qMatmulSIMD(Q_matrix8_t* A, Q_matrix8_t* B, Q_matrix32_t* acc, Q_matrix8_t* output){

}
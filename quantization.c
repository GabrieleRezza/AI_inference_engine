#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include "matrix.h"
#include "quantization.h"
#include <assert.h>
#define Q_PAR_THRESHOLD 65536   
#define REQ_PAR_THRESHOLD 65536   

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


void CleanInt8Mat(Q_matrix8_t *mat){
    int dim = mat->cols * mat->rows, i;

    if (mat == NULL){
        printf("Null matrix input. \n\n");
        return;
    }

    for(i = 0; i < dim; i++)
        mat->data_8[i] = (int8_t) 0;
}

void CleanInt32Mat(Q_matrix32_t *mat){
    int dim = mat->cols * mat->rows, i;

    if (mat == NULL){
        printf("Null matrix input. \n\n");
        return;
    }

    for(i = 0; i < dim; i++)
        mat->data_32[i] = (int32_t) 0;
}

static inline float hmax_ps_avx2(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 max = _mm_max_ps(lo, hi);

    max = _mm_max_ps(max, _mm_shuffle_ps(max, max, _MM_SHUFFLE(1, 0, 3, 2)));
    max = _mm_max_ps(max, _mm_shuffle_ps(max, max, _MM_SHUFFLE(2, 3, 0, 1)));

    return _mm_cvtss_f32(max);
}



int quantizeSIMD(const matrix_t *input, Q_matrix8_t *mat8, float qmax) {
    if (input == NULL || mat8 == NULL || input->rows == 0 || input->cols == 0) {
        printf("Empty matrix \n");
        return -1;
    }
    assert(mat8->rows == input->rows && mat8->cols == input->cols);

    const int mat_dim = input->rows * input->cols;
    const int vec_end = mat_dim & ~7;
    const int par = (mat_dim >= Q_PAR_THRESHOLD);
    const float *data = input->data;
    float max_abs = 0.0f;

    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    #pragma omp parallel if(par)
    {
        __m256 vmax = _mm256_setzero_ps();

        #pragma omp for nowait
        for (int i = 0; i < vec_end; i += 8) {
            __m256 a = _mm256_and_ps(_mm256_loadu_ps(&data[i]), sign_mask);
            vmax = _mm256_max_ps(vmax, a);
        }

        float t = hmax_ps_avx2(vmax);
        #pragma omp critical
        { if (t > max_abs) max_abs = t; }
    }

    for (int i = vec_end; i < mat_dim; i++) {
        float a = fabsf(data[i]);
        if (a > max_abs) max_abs = a;
    }

    if (!isfinite(max_abs)) {          
        printf("Non-finite values in input \n");
        return -1;
    }

    if (max_abs == 0.0f) {
        mat8->scale_factor = 1.0f;
    } else {
        mat8->scale_factor = qmax / max_abs;
        if (!isfinite(mat8->scale_factor)) {   
            printf("Invalid scale factor \n");
            return -1;
        }
    }

    const float scale = mat8->scale_factor;
    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256 v_hi    = _mm256_set1_ps(qmax);
    const __m256 v_lo    = _mm256_set1_ps(-qmax);

    #pragma omp parallel for schedule(static) if(par)
    for (int i = 0; i < vec_end; i += 8) {
        __m256 f = _mm256_mul_ps(_mm256_loadu_ps(&data[i]), v_scale);
        f = _mm256_min_ps(_mm256_max_ps(f, v_lo), v_hi);
        __m256i r32 = _mm256_cvtps_epi32(f);      /* round-to-nearest-even */

        __m128i v16 = _mm_packs_epi32(_mm256_castsi256_si128(r32),
                                      _mm256_extracti128_si256(r32, 1));
        __m128i v8  = _mm_packs_epi16(v16, _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)&mat8->data_8[i], v8);
    }

    for (int i = vec_end; i < mat_dim; i++) {
        float t = data[i] * scale;
        if (t > qmax)       t = qmax;
        else if (t < -qmax) t = -qmax;
        mat8->data_8[i] = (int8_t)lrintf(t);
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
    else {
        dequantize_factor = 1.0f / input->scale_factor;

    }

    for(i = 0; i < mat_dim; i++){
        output->data[i] = (float) input->data_8[i] * dequantize_factor;
    }
    return 0;
}

int dequantizeSIMD(const Q_matrix8_t *input, matrix_t *output) {
    int i, mat_dim;
    float dequantize_factor;

    if (input == NULL || output == NULL || input->rows == 0 || input->cols == 0) {
        printf("Invalid input\n");
        return -1;
    }

    output->rows = input->rows;
    output->cols = input->cols;
    mat_dim = input->cols * input->rows;
    
    if (input->scale_factor == 0.0f) {
        dequantize_factor = 1.0f;
    } else {
        dequantize_factor = 1.0f / input->scale_factor;
    }

    __m256 v_factor = _mm256_set1_ps(dequantize_factor);

    #pragma omp parallel for schedule(static)
    for (i = 0; i <= mat_dim - 8; i += 8) {
        /* Load 8 * int8 */
        __m128i v_in_8 = _mm_loadl_epi64((const __m128i*)&input->data_8[i]);

        /*Sign-extension: 8 x int8 -> 8 x int32 su __m256i */
        __m256i v_in_32 = _mm256_cvtepi8_epi32(v_in_8);

        /* Int ---> float */
        __m256 v_f32 = _mm256_cvtepi32_ps(v_in_32);

        /* Dequantizing */
        __m256 v_res = _mm256_mul_ps(v_f32, v_factor);

        _mm256_storeu_ps(&output->data[i], v_res);
    }

    for (i = i; i < mat_dim; i++) {
        output->data[i] = (float)input->data_8[i] * dequantize_factor;
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
    int A_dim, B_dim, block_i, block_j, block_k, min_i, min_j, min_k, i, j, k;
    int B_column_sum [B->cols], stop_sum;
    
    if(A == NULL || B == NULL ||acc == NULL || A->data_8 == NULL || B->data_8 == NULL || acc->data_32 == NULL || A->cols != B->rows){
        printf("Invalid matrices \n\n");
        return -1;
    }

    A_dim = A->cols * A->rows;
    B_dim = B->cols * B->rows;

    output->rows = A->rows;
    output->cols = B->cols;
    
    //The output matrix does not need to be cleaned since it will be overwritten
    CleanInt32Mat(acc);


    for(i = 0; i < B->cols; i++)
        B_column_sum[i] = 0;
    
    omp_set_num_threads(omp_get_max_threads());

    stop_sum = B->cols - (B->cols % 32); //if b->cols < 32 --> stop_sum = 0

    if(B->cols >= 32){

        #pragma omp parallel for private(j) schedule(static)
        {
            for(i = 0; i < stop_sum; i+=32){
                

                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                __m256i acc4 = _mm256_setzero_si256();
                for(j = 0; j < B->rows; j++){

                    __m256i load = _mm256_loadu_si256((__m256i const*)&B->data_8[j*B->cols + i]);

                    __m128i lo = _mm256_castsi256_si128(load);
                    __m128i hi = _mm256_extracti128_si256(load, 1);

                    __m256i temp = _mm256_cvtepi8_epi32(lo);//1-8
                    acc1 = _mm256_add_epi32(acc1, temp);

                    temp = _mm256_cvtepi8_epi32(_mm_bsrli_si128(lo, 8)); //9-16
                    acc2 = _mm256_add_epi32(acc2, temp);

                    temp = _mm256_cvtepi8_epi32(hi); //17-24
                    acc3 = _mm256_add_epi32(acc3, temp);

                    temp = _mm256_cvtepi8_epi32(_mm_bsrli_si128(hi, 8)); //25-32
                    acc4 = _mm256_add_epi32(acc4, temp);
                    
                }
                _mm256_storeu_si256((__m256i *)&B_column_sum[i], acc1);
                _mm256_storeu_si256((__m256i *)&B_column_sum[i + 8], acc2);
                _mm256_storeu_si256((__m256i *)&B_column_sum[i + 16], acc3);
                _mm256_storeu_si256((__m256i *)&B_column_sum[i + 24], acc4);
            }
        }
    }

    if (stop_sum < B->cols){
        for(i = 0; i < B->rows; i++){
            for(j = stop_sum; j < B->cols; j++){
                B_column_sum[j] += B->data_8[i*B->cols + j];
            }
        }
    }

    #pragma omp parallel
    {
        for(block_i = 0; block_i < output->rows; block_i += BLOCK_SIZE){
            for(block_j = 0; block_j < output->cols; block_j += BLOCK_SIZE){
                for(block_k = 0; block_k < B->rows; block_k += BLOCK_SIZE){
                    min_i = min(block_i + BLOCK_SIZE, output->rows);
                    min_j = min(block_j + BLOCK_SIZE, output->cols);
                    min_k = min(block_k + BLOCK_SIZE, B->rows);


                    
                    for(i = block_i; i < min_i; i++){
                        for(j = block_j; j < min_j - (min_j % 8); j += 8){

                            //ATTENZIONE qui devo aggiungere a ogni ra in ingresso 127 per usare maddubs 
                            //POI a fine accumulazione per ogni elemento tolgo lo shift: -127 * sommacolonnaB_j
                            //tutto prima di passare a requantize
                            __m256i r_acc = _mm256_loadu_si256((const __m256i *)&acc->data_32[i*output->cols + j]);

                            for(k = block_k; k < min_k; k++){

                                __m256i ra8 = _mm256_set1_epi8(A->data_8[i*A->cols + k]);
                                    
                                __m256i rb8 = _mm256_loadu_si256((const __m256i *)&B->data_8[k*B->cols + j]);
                                
                                //ora ho il vettore dove accumulare e i due vettori che vanno moltiplicati

                                
                            }

                            //store

                        }
                    }
                    
                    //Handling the remaining columns that are not a multiple of 8 outside of the parallelized loop
                    
                }
            }
        }
    }
}
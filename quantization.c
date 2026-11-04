#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include "matrix.h"
#include "quantization.h"
#define Q_PAR_THRESHOLD 65536
#define REQ_PAR_THRESHOLD 65536
#define DEQ_PAR_THRESHOLD 65536
//Column tile of the quantized matmul: smaller than BLOCK_SIZE so that vector x matrix products still create enough tasks for every thread
#define QBLOCK_J 64

_Static_assert(BLOCK_SIZE % 4 == 0, "BLOCK_SIZE deve essere un multiplo di 4");
_Static_assert(QBLOCK_J % 8 == 0, "QBLOCK_J deve essere un multiplo di 8");

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
    if (mat == NULL){
        printf("Null matrix input. \n\n");
        return;
    }

    memset(mat->data_8, 0, sizeof(int8_t) * mat->cols * mat->rows);
}

void CleanInt32Mat(Q_matrix32_t *mat){
    if (mat == NULL){
        printf("Null matrix input. \n\n");
        return;
    }

    memset(mat->data_32, 0, sizeof(int32_t) * mat->cols * mat->rows);
}

static inline float hmax_ps_avx2(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 max = _mm_max_ps(lo, hi);

    max = _mm_max_ps(max, _mm_shuffle_ps(max, max, _MM_SHUFFLE(1, 0, 3, 2)));
    max = _mm_max_ps(max, _mm_shuffle_ps(max, max, _MM_SHUFFLE(2, 3, 0, 1)));

    return _mm_cvtss_f32(max);
}



int quantize(const matrix_t *input, Q_matrix8_t *mat8, float qmax) {
    if (input == NULL || mat8 == NULL || input->data == NULL || mat8->data_8 == NULL || input->rows <= 0 || input->cols <= 0) {
        printf("Empty matrix \n");
        return -1;
    }
    if (!(qmax > 0.0f && qmax <= 127.0f)) {
        printf("Invalid qmax \n");
        return -1;
    }

    //The caller guarantees that mat8 has enough memory (rows*cols), the dimensions are copied from the input
    mat8->rows = input->rows;
    mat8->cols = input->cols;

    const int mat_dim = input->rows * input->cols;
    const int vec_end = mat_dim & ~7;
    const int par = (mat_dim >= Q_PAR_THRESHOLD);
    const float *data = input->data;
    float max_abs = 0.0f;
    int has_nan = 0;

    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    #pragma omp parallel if(par)
    {
        __m256 vmax = _mm256_setzero_ps();
        __m256 vnan = _mm256_setzero_ps();

        #pragma omp for nowait
        for (int i = 0; i < vec_end; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 a = _mm256_and_ps(x, sign_mask);
            //max_ps drops NaN values, so they are tracked separately (unordered compare is true only for NaN)
            vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
            vmax = _mm256_max_ps(vmax, a);
        }

        float t = hmax_ps_avx2(vmax);
        int t_nan = _mm256_movemask_ps(vnan) != 0;
        #pragma omp critical
        {
            if (t > max_abs) max_abs = t;
            if (t_nan) has_nan = 1;
        }
    }

    for (int i = vec_end; i < mat_dim; i++) {
        float a = fabsf(data[i]);
        if (isnan(a)) has_nan = 1;
        if (a > max_abs) max_abs = a;
    }

    if (has_nan || !isfinite(max_abs)) {
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

    if( input == NULL || output == NULL || input->rows <= 0 || input->cols <= 0){
        printf("Invalid input \n");
        return -1;
    }
    output->rows = input->rows;
    output->cols = input->cols;
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
    int mat_dim, vec_end, par;
    float dequantize_factor;

    if (input == NULL || output == NULL || input->rows <= 0 || input->cols <= 0) {
        printf("Invalid input\n");
        return -1;
    }

    output->rows = input->rows;
    output->cols = input->cols;
    mat_dim = input->cols * input->rows;
    //The tail must start from a fixed index: the loop variable of an omp for is undefined after the loop
    vec_end = mat_dim & ~7;
    par = (mat_dim >= DEQ_PAR_THRESHOLD);

    if (input->scale_factor == 0.0f) {
        dequantize_factor = 1.0f;
    } else {
        dequantize_factor = 1.0f / input->scale_factor;
    }

    __m256 v_factor = _mm256_set1_ps(dequantize_factor);

    #pragma omp parallel for schedule(static) if(par)
    for (int i = 0; i < vec_end; i += 8) {
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

    for (int i = vec_end; i < mat_dim; i++) {
        output->data[i] = (float)input->data_8[i] * dequantize_factor;
    }

    return 0;
}



int requantize(Q_matrix32_t* input, Q_matrix8_t* output, float max_value){
    int i, mat_dim;
    int32_t abs_value, max_abs = 0;
    float temp, rescale;

    if(input == NULL || output == NULL || input->rows <= 0 || input->cols <= 0){
        printf("Empty matrix \n");
        return -1;
    }

    mat_dim = input->cols * input->rows;
    output->rows = input->rows;
    output->cols = input->cols;


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
        temp = (float)input->data_32[i] * rescale;

        if (temp > max_value)
            temp = max_value;
        else if(temp < -max_value)
            temp = -max_value;

        //lrintf rounds half to even, exactly like _mm256_cvtps_epi32 in the SIMD version
        output->data_8[i] = (int8_t)lrintf(temp);
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
    int32_t max_abs = 0;
    float rescale;

    //The checks must come before any access to input/output
    if (input == NULL || output == NULL || input->data_32 == NULL || output->data_8 == NULL || input->rows <= 0 || input->cols <= 0) {
        printf("Empty or invalid matrix\n");
        return -1;
    }
    if (!(input->scale_factor > 0.0f)) {      /* intercetta 0, negativi e NaN */
        printf("Invalid input scale factor\n");
        return -1;
    }
    if (!(max_value > 0.0f && max_value <= 127.0f)) {
        printf("Invalid max_value\n");
        return -1;
    }

    const int mat_dim = input->rows * input->cols;
    const int vec_end = mat_dim & ~7;         //cuts off the last 7 elements if needed
    const int par = (mat_dim >= REQ_PAR_THRESHOLD);

    //The caller guarantees that output has enough memory (rows*cols)
    output->rows = input->rows;
    output->cols = input->cols;

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



/*Int8 x int8 matmul accumulated in int32.
  maddubs needs an unsigned first operand: instead of shifting A by +127 (which makes a pair of products reach
  254*127*2 = 64516 and saturate the int16 result of maddubs) the sign of A is moved onto B:
        a*b = |a| * (b * sign(a))
  With |a| <= 127 and |b| <= 127 a pair is at most 127*127*2 = 32258 < 32767, so maddubs never saturates
  and no column-sum correction is needed.
  Requires values in [-127, 127] (guaranteed by quantize/requantize, which clamp to +-qmax <= 127). */
int qMatmulAcc32(const Q_matrix8_t *A, const Q_matrix8_t *B, Q_matrix32_t *acc, const Q_matrix32_t *bias){
    int block_i, block_j, rows, cols, inner;

    if(A == NULL || B == NULL || acc == NULL || A->data_8 == NULL || B->data_8 == NULL || acc->data_32 == NULL){
        printf("Invalid matrices \n\n");
        return -1;
    }
    if(A->rows <= 0 || A->cols <= 0 || B->cols <= 0 || A->cols != B->rows){
        printf("Multiplication impossible, wrong dimensions. \n\n");
        return -1;
    }
    if(bias != NULL && (bias->data_32 == NULL || bias->rows * bias->cols != B->cols)){
        printf("Invalid bias \n\n");
        return -1;
    }

    rows = A->rows;
    cols = B->cols;
    inner = B->rows;

    //The caller guarantees that acc has enough memory (A->rows * B->cols)
    acc->rows = rows;
    acc->cols = cols;
    acc->scale_factor = A->scale_factor * B->scale_factor;

    const int8_t *a_data = A->data_8;
    const int8_t *b_data = B->data_8;
    int32_t *c_data = acc->data_32;
    const int32_t *bias_data = (bias != NULL) ? bias->data_32 : NULL;

    //Every thread owns a different (block_i, block_j) tile of acc and goes through the whole k dimension: no races
    #pragma omp parallel for collapse(2) schedule(static)
    for (block_i = 0; block_i < rows; block_i += BLOCK_SIZE) {
        for (block_j = 0; block_j < cols; block_j += QBLOCK_J) {

            const int min_i = min(block_i + BLOCK_SIZE, rows);
            const int min_j = min(block_j + QBLOCK_J, cols);
            const int j_simd_limit = block_j + ((min_j - block_j) & ~7);
            const __m256i ones_16 = _mm256_set1_epi16(1);

            //The tile is initialized with the bias (or 0), so no CleanInt32Mat is needed
            for (int i = block_i; i < min_i; i++) {
                for (int j = block_j; j < min_j; j++)
                    c_data[i * cols + j] = (bias_data != NULL) ? bias_data[j] : 0;
            }

            for (int block_k = 0; block_k < inner; block_k += BLOCK_SIZE) {
                const int min_k = min(block_k + BLOCK_SIZE, inner);
                const int k_simd_limit = block_k + ((min_k - block_k) & ~3);

                for (int i = block_i; i < min_i; i++) {
                    const int8_t *a_row = &a_data[i * inner];
                    int k, j;

                    // --- CICLO VETTORIALE (8 colonne x 4 righe di k) ---
                    for (j = block_j; j < j_simd_limit; j += 8) {

                        __m256i r_acc = _mm256_loadu_si256((const __m256i *)&c_data[i * cols + j]);

                        for (k = block_k; k < k_simd_limit; k += 4) {
                            // Carichiamo 8 colonne per 4 righe consecutive di B
                            __m128i b0 = _mm_loadl_epi64((const __m128i*)&b_data[(k + 0) * cols + j]);
                            __m128i b1 = _mm_loadl_epi64((const __m128i*)&b_data[(k + 1) * cols + j]);
                            __m128i b2 = _mm_loadl_epi64((const __m128i*)&b_data[(k + 2) * cols + j]);
                            __m128i b3 = _mm_loadl_epi64((const __m128i*)&b_data[(k + 3) * cols + j]);

                            // Intrecciamo a livello di byte: [b0, b1, b0, b1...] e [b2, b3, b2, b3...]
                            __m128i b01 = _mm_unpacklo_epi8(b0, b1);
                            __m128i b23 = _mm_unpacklo_epi8(b2, b3);

                            // Intrecciamo a 16-bit per ottenere quartetti contigui: [b0, b1, b2, b3] per ogni colonna
                            __m128i b_lo = _mm_unpacklo_epi16(b01, b23); // Colonne j+0 .. j+3
                            __m128i b_hi = _mm_unpackhi_epi16(b01, b23); // Colonne j+4 .. j+7
                            __m256i rb_quads = _mm256_inserti128_si256(_mm256_castsi128_si256(b_lo), b_hi, 1);

                            // I 4 scalari consecutivi di A sono gia' nell'ordine giusto in memoria: [a0, a1, a2, a3]
                            int32_t a_quad;
                            memcpy(&a_quad, &a_row[k], sizeof(int32_t));
                            __m256i ra_quads = _mm256_set1_epi32(a_quad);

                            // |a| come uint8 e b * sign(a) come int8 (sign_epi8 azzera anche b dove a == 0)
                            __m256i a_abs  = _mm256_sign_epi8(ra_quads, ra_quads);
                            __m256i b_sign = _mm256_sign_epi8(rb_quads, ra_quads);

                            // Moltiplica uint8*int8 e somma a coppie in int16 (16 parziali, nessuna saturazione)
                            __m256i prod_16 = _mm256_maddubs_epi16(a_abs, b_sign);
                            // Somma le coppie di int16 adiacenti in int32 (8 risultati finali, uno per colonna)
                            __m256i prod_32 = _mm256_madd_epi16(prod_16, ones_16);

                            r_acc = _mm256_add_epi32(r_acc, prod_32);
                        }

                        // Residui di k (solo se B->rows non e' multiplo di 4 sull'ultimo blocco)
                        for (; k < min_k; k++) {
                            __m256i ra = _mm256_set1_epi32((int32_t)a_row[k]);
                            __m128i b_raw = _mm_loadl_epi64((const __m128i*)&b_data[k * cols + j]);
                            __m256i rb = _mm256_cvtepi8_epi32(b_raw);
                            r_acc = _mm256_add_epi32(r_acc, _mm256_mullo_epi32(ra, rb));
                        }

                        _mm256_storeu_si256((__m256i *)&c_data[i * cols + j], r_acc);
                    }

                    // --- Last columns, not multiple of 8 ---
                    for (j = j_simd_limit; j < min_j; j++) {
                        int32_t sum = c_data[i * cols + j];
                        for (k = block_k; k < min_k; k++) {
                            sum += (int32_t)a_row[k] * (int32_t)b_data[k * cols + j];
                        }
                        c_data[i * cols + j] = sum;
                    }
                }
            }
        }
    }

    return 0;
}



int qMatmulSIMD(const Q_matrix8_t* A, const Q_matrix8_t* B, Q_matrix32_t* acc, Q_matrix8_t* output, const Q_matrix32_t *bias){
    if(output == NULL || output->data_8 == NULL){
        printf("Invalid output matrix \n\n");
        return -1;
    }

    if(qMatmulAcc32(A, B, acc, bias) != 0)
        return -1;

    // L'accumulatore contiene i valori matematicamente corretti in int32 con scala sA * sB
    return requantizeSIMD(acc, output, 127.0f);
}



int dequantizeAcc32(const Q_matrix32_t *acc, const matrix_t *bias, matrix_t *output, int relu){
    int rows, cols, j_simd_limit, par;
    float dequantize_factor;

    if(acc == NULL || output == NULL || acc->data_32 == NULL || output->data == NULL || acc->rows <= 0 || acc->cols <= 0){
        printf("Invalid input\n");
        return -1;
    }
    if(!(acc->scale_factor > 0.0f)){
        printf("Invalid input scale factor\n");
        return -1;
    }
    if(bias != NULL && (bias->data == NULL || bias->rows * bias->cols != acc->cols)){
        printf("Invalid bias\n");
        return -1;
    }

    rows = acc->rows;
    cols = acc->cols;
    j_simd_limit = cols & ~7;
    par = (rows * cols >= DEQ_PAR_THRESHOLD);
    dequantize_factor = 1.0f / acc->scale_factor;

    //The caller guarantees that output has enough memory (rows*cols)
    output->rows = rows;
    output->cols = cols;

    const __m256 v_factor = _mm256_set1_ps(dequantize_factor);
    const __m256 v_zero = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < rows; i++){
        const int32_t *in_row = &acc->data_32[i * cols];
        float *out_row = &output->data[i * cols];
        int j;

        for(j = 0; j < j_simd_limit; j += 8){
            __m256 f = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)&in_row[j]));
            f = _mm256_mul_ps(f, v_factor);
            if(bias != NULL)
                f = _mm256_add_ps(f, _mm256_loadu_ps(&bias->data[j]));
            if(relu)
                f = _mm256_max_ps(f, v_zero);
            _mm256_storeu_ps(&out_row[j], f);
        }

        for(; j < cols; j++){
            float f = (float)in_row[j] * dequantize_factor;
            if(bias != NULL)
                f += bias->data[j];
            if(relu && f < 0.0f)
                f = 0.0f;
            out_row[j] = f;
        }
    }

    return 0;
}

#include "../matrix.h"
#include "../network.h"
#include "../quantization.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

static int failures = 0;

#define CHECK(cond, ...) do { if(!(cond)){ printf("  FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while(0)

static void fill_q8(Q_matrix8_t *m, int extremes){
    int i;
    for(i = 0; i < m->rows * m->cols; i++){
        if(extremes)
            m->data_8[i] = (rand() & 1) ? 127 : -127;   //worst case for the int16 saturation of maddubs
        else
            m->data_8[i] = (int8_t)(rand() % 255 - 127);
    }
}

//Naive int32 reference
static void ref_matmul(const Q_matrix8_t *A, const Q_matrix8_t *B, const Q_matrix32_t *bias, int32_t *out){
    int i, j, k;
    for(i = 0; i < A->rows; i++)
        for(j = 0; j < B->cols; j++){
            int32_t s = bias ? bias->data_32[j] : 0;
            for(k = 0; k < A->cols; k++)
                s += (int32_t)A->data_8[i*A->cols + k] * (int32_t)B->data_8[k*B->cols + j];
            out[i*B->cols + j] = s;
        }
}

static void test_qmatmul(int M, int K, int N, int extremes, int use_bias){
    Q_matrix8_t *A = createMatrix8(M, K), *B = createMatrix8(K, N);
    Q_matrix32_t *acc = createMatrix32(M, N), *bias = createMatrix32(1, N);
    int32_t *ref = malloc(sizeof(int32_t) * M * N);
    int i, bad = 0;

    fill_q8(A, extremes);
    fill_q8(B, extremes);
    for(i = 0; i < N; i++)
        bias->data_32[i] = rand() % 20001 - 10000;

    ref_matmul(A, B, use_bias ? bias : NULL, ref);
    CHECK(qMatmulAcc32(A, B, acc, use_bias ? bias : NULL) == 0, "qMatmulAcc32 returned error");
    for(i = 0; i < M * N; i++)
        if(acc->data_32[i] != ref[i]) bad++;

    printf("qMatmulAcc32 %4dx%-4d * %4dx%-4d extremes=%d bias=%d : %s\n", M, K, K, N, extremes, use_bias, bad ? "FAIL" : "ok");
    CHECK(bad == 0, "%d wrong elements", bad);

    free(ref);
    deleteMatQ8(A); deleteMatQ8(B); deleteMatQ32(acc); deleteMatQ32(bias);
}

static void test_quant_roundtrip(int n){
    matrix_t *x = CreateMatrix(1, n), *y = CreateMatrix(1, n);
    Q_matrix8_t *q = createMatrix8(1, n);
    float max_err = 0.0f;
    int i;

    for(i = 0; i < n; i++)
        x->data[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;

    CHECK(quantize(x, q, 127.0f) == 0, "quantize error");
    CHECK(dequantizeSIMD(q, y) == 0, "dequantizeSIMD error");
    for(i = 0; i < n; i++){
        float e = fabsf(x->data[i] - y->data[i]);
        if(e > max_err) max_err = e;
    }
    //Theoretical bound: half a quantization step (small tolerance for float rounding)
    float bound = 0.5f / q->scale_factor * 1.001f;
    printf("quantize/dequantizeSIMD n=%-6d max_err=%f bound=%f : %s\n", n, max_err, bound, max_err <= bound ? "ok" : "FAIL");
    CHECK(max_err <= bound, "roundtrip error too big");

    DeleteMatrix(x); DeleteMatrix(y); deleteMatQ8(q);
}

static int build_random_network(network_t *net, const int *sizes, int n_layers){
    int i, j;
    net->layer_quantity = n_layers;
    net->max_layer_size = 0;
    net->layers = _mm_malloc(sizeof(layer_t) * n_layers, 32);
    if(net->layers == NULL) return -1;

    for(i = 0; i < n_layers; i++){
        layer_t *l = &net->layers[i];
        l->input_size = sizes[i];
        l->output_size = sizes[i + 1];
        if(l->input_size > net->max_layer_size) net->max_layer_size = l->input_size;
        if(l->output_size > net->max_layer_size) net->max_layer_size = l->output_size;
        l->weights = CreateMatrix(l->input_size, l->output_size);
        l->biases = CreateMatrix(1, l->output_size);
        if(l->weights == NULL || l->biases == NULL) return -1;
        //Xavier-like init so that the activations keep a reasonable range
        float r = sqrtf(6.0f / (l->input_size + l->output_size));
        for(j = 0; j < l->input_size * l->output_size; j++)
            l->weights->data[j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * r;
        for(j = 0; j < l->output_size; j++)
            l->biases->data[j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    return 0;
}

static void test_forward(const int *sizes, int n_layers, int iterations){
    network_t net;
    q_network_t qnet;
    matrix_t *input, *out_f, *out_q;
    double num = 0.0, den = 0.0, t0, t_f, t_q;
    int i;

    if(build_random_network(&net, sizes, n_layers) != 0 || QuantizeNetwork(&net, &qnet) != 0){
        CHECK(0, "network creation failed");
        return;
    }

    input = CreateMatrix(1, sizes[0]);
    for(i = 0; i < sizes[0]; i++)
        input->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    out_f = ForwardPass(&net, input);
    out_q = QForwardPass(&qnet, input);
    CHECK(out_f != NULL && out_q != NULL, "forward pass returned NULL");
    if(out_f != NULL && out_q != NULL){
        CHECK(out_f->cols == out_q->cols && out_q->cols == sizes[n_layers], "wrong output size");
        for(i = 0; i < out_f->cols; i++){
            double d = out_f->data[i] - out_q->data[i];
            num += d * d;
            den += (double)out_f->data[i] * out_f->data[i];
        }
    }
    double rel = sqrt(num / (den > 0 ? den : 1));

    t0 = omp_get_wtime();
    for(i = 0; i < iterations; i++) DeleteMatrix(ForwardPass(&net, input));
    t_f = (omp_get_wtime() - t0) / iterations;
    t0 = omp_get_wtime();
    for(i = 0; i < iterations; i++) DeleteMatrix(QForwardPass(&qnet, input));
    t_q = (omp_get_wtime() - t0) / iterations;

    printf("Forward %d layers (in %d, max %d): relative error = %.4f%%  float %.6f s  quant %.6f s : %s\n",
           n_layers, sizes[0], net.max_layer_size, rel * 100.0, t_f, t_q, rel < 0.03 ? "ok" : "FAIL");
    CHECK(rel < 0.03, "quantized output too far from float output");

    DeleteMatrix(input); DeleteMatrix(out_f); DeleteMatrix(out_q);
    FreeNetPartial(&net, net.layer_quantity);
    _mm_free(net.layers);
    FreeQNetwork(&qnet);
}

int main(){
    int sizes_list[][3] = { {1,4,8}, {1,37,45}, {3,130,259}, {1,512,512}, {7,129,1}, {130,257,131}, {256,256,256} };
    int i, n;

    srand(1234);

    for(i = 0; i < (int)(sizeof(sizes_list) / sizeof(sizes_list[0])); i++){
        test_qmatmul(sizes_list[i][0], sizes_list[i][1], sizes_list[i][2], 1, 0);
        test_qmatmul(sizes_list[i][0], sizes_list[i][1], sizes_list[i][2], 0, 1);
    }

    for(n = 1; n <= 17; n++)
        test_quant_roundtrip(n);
    test_quant_roundtrip(100000);   //above the threading threshold

    {
        int small[] = {2, 3, 2};
        int big[] = {784, 512, 256, 10};      //first input larger than every output: old max_layer_size overflow
        int mid[] = {300, 1000, 37, 129};
        test_forward(small, 2, 1000);
        test_forward(big, 3, 200);
        test_forward(mid, 3, 200);
    }

    printf("\n%s (%d failures)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures);
    return failures != 0;
}

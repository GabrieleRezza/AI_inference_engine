#include "../matrix.h"
#include "../network.h"
#include "../quantization.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void bench_matmul(int M, int K, int N, int it){
    matrix_t *A = CreateMatrix(M,K), *B = CreateMatrix(K,N), *C = CreateMatrix(M,N);
    Q_matrix8_t *qA = createMatrix8(M,K), *qB = createMatrix8(K,N);
    Q_matrix32_t *acc = createMatrix32(M,N);
    InitMatrixRandom(A); InitMatrixRandom(B);
    quantize(A,qA,127.0f); quantize(B,qB,127.0f);
    double tf = BenchmarkMatMul(A,B,C,it), tq = BenchmarkQMatMul(qA,qB,acc,it);
    printf("| %4dx%-4d * %4dx%-4d | %10.6f | %10.6f | %5.2fx |\n", M,K,K,N, tf, tq, tf/tq);
    DeleteMatrix(A); DeleteMatrix(B); DeleteMatrix(C); deleteMatQ8(qA); deleteMatQ8(qB); deleteMatQ32(acc);
}

static void bench_forward(const int *s, int n, int it){
    network_t net; q_network_t q; int i,j;
    net.layer_quantity=n; net.max_layer_size=0; net.layers=_mm_malloc(sizeof(layer_t)*n,32);
    for(i=0;i<n;i++){
        layer_t *l=&net.layers[i]; l->input_size=s[i]; l->output_size=s[i+1];
        if(s[i]>net.max_layer_size) net.max_layer_size=s[i];
        if(s[i+1]>net.max_layer_size) net.max_layer_size=s[i+1];
        l->weights=CreateMatrix(s[i],s[i+1]); l->biases=CreateMatrix(1,s[i+1]);
        InitMatrixRandom(l->weights); for(j=0;j<s[i+1];j++) l->biases->data[j]=0.01f;
    }
    if(QuantizeNetwork(&net,&q)) return;
    matrix_t *in=CreateMatrix(1,s[0]); InitMatrixRandom(in);
    double tf=BenchmarkForward(&net,NULL,in,it,0), tq=BenchmarkForward(NULL,&q,in,it,1);
    printf("| ");
    for(i=0;i<=n;i++) printf("%d%s",s[i],i<n?"-":"");
    printf(" | %10.6f | %10.6f | %5.2fx |\n", tf, tq, tf/tq);
    DeleteMatrix(in); FreeNetPartial(&net,n); _mm_free(net.layers); FreeQNetwork(&q);
}

int main(void){
    int sz[][3] = {{1,512,512},{1,2048,2048},{64,512,512},{256,256,256},{512,512,512},{1024,1024,1024}};
    int a[]={784,512,256,10}, b[]={1024,2048,2048,1024}, c[]={4096,4096,4096};
    printf("Matmul (seconds, avg per op)\n| size | float | int8 | speedup |\n|---|---|---|---|\n");
    for(int i=0;i<6;i++) bench_matmul(sz[i][0],sz[i][1],sz[i][2], sz[i][0]*sz[i][1]*sz[i][2] > 200000000 ? 5 : 50);
    printf("\nForward pass 1 x n (seconds, avg per pass)\n| layers | float | quantized | speedup |\n|---|---|---|---|\n");
    bench_forward(a,3,200); bench_forward(b,3,50); bench_forward(c,2,10);
    return 0;
}

#include "../matrix.h"
#include "../network.h"
#include "../quantization.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if(!(cond)){ printf("  FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while(0)

//Hand computed 2x3 * 3x2 : [[1,-2,3],[0,127,-127]] * [[4,5],[-6,7],[8,-9]] = [[40,-36],[-1778... computed below
static void test_hand(void){
    Q_matrix8_t *A = createMatrix8(2,3), *B = createMatrix8(3,2), *out = createMatrix8(2,2);
    Q_matrix32_t *acc = createMatrix32(2,2), *bias = createMatrix32(1,2);
    int8_t a[] = {1,-2,3, 0,127,-127}, b[] = {4,5, -6,7, 8,-9};
    //row0: 1*4+-2*-6+3*8 = 40 ; 1*5+-2*7+3*-9 = -36 ; row1: 127*-6+-127*8 = -1778 ; 127*7+-127*-9 = 2032
    int32_t exp[] = {40,-36,-1778,2032}, expb[] = {140,-136+100,-1678,2132};

    memcpy(A->data_8,a,6); memcpy(B->data_8,b,6);
    A->scale_factor = 2.0f; B->scale_factor = 3.0f;
    bias->data_32[0] = 100; bias->data_32[1] = 0;
    CHECK(qMatmulAcc32(A,B,acc,NULL)==0,"err");

    for(int i=0;i<4;i++) CHECK(acc->data_32[i]==exp[i],"hand nobias [%d]=%d exp %d",i,acc->data_32[i],exp[i]);
    CHECK(acc->scale_factor==6.0f,"acc scale %f",acc->scale_factor);
    bias->data_32[1] = 0;
    CHECK(qMatmulAcc32(A,B,acc,bias)==0,"err");

    int32_t eb[] = {140,-36,-1778+100,2032};   //bias is per column: col0 +100, col1 +0

    for(int i=0;i<4;i++) CHECK(acc->data_32[i]==eb[i],"hand bias [%d]=%d exp %d",i,acc->data_32[i],eb[i]);
    (void)expb;
    //qMatmulSIMD: max |acc| = 2032 -> rescale 127/2032 ; 2032 -> 127, -1778 -> -111.1 -> -111, 40 -> 2.5 -> 2 (ties to even), -36 -> -2.25 -> -2
    CHECK(qMatmulSIMD(A,B,acc,out,NULL)==0,"err");

    int8_t eo[] = {2,-2,-111,127};
    for(int i=0;i<4;i++) CHECK(out->data_8[i]==eo[i],"qMatmulSIMD [%d]=%d exp %d",i,out->data_8[i],eo[i]);
    printf("hand-computed 2x3*3x2 + bias + qMatmulSIMD: %s\n", failures? "FAIL":"ok");
    deleteMatQ8(A);deleteMatQ8(B);deleteMatQ8(out);deleteMatQ32(acc);deleteMatQ32(bias);
}

static void test_uniform(int v_a, int v_b, int K, const char *name){
    Q_matrix8_t *A = createMatrix8(5,K), *B = createMatrix8(K,19);
    Q_matrix32_t *acc = createMatrix32(5,19);
    int bad = 0;
    memset(A->data_8, v_a, 5*K); memset(B->data_8, v_b, K*19);

    for(int i=0;i<5*K;i++) A->data_8[i]=(int8_t)v_a;
    for(int i=0;i<K*19;i++) B->data_8[i]=(int8_t)v_b;

    qMatmulAcc32(A,B,acc,NULL);
    for(int i=0;i<5*19;i++) if(acc->data_32[i]!=v_a*v_b*K) bad++;

    printf("%-28s K=%-6d exp %d : %s\n", name, K, v_a*v_b*K, bad?"FAIL":"ok");
    CHECK(bad==0,"%s: %d wrong",name,bad);
    deleteMatQ8(A);deleteMatQ8(B);deleteMatQ32(acc);
}

static void test_requant_zero(void){
    Q_matrix32_t *acc = createMatrix32(3,5); Q_matrix8_t *o = createMatrix8(3,5);
    memset(acc->data_32,0,15*4); acc->scale_factor = 4.0f;
    CHECK(requantizeSIMD(acc,o,127.0f)==0,"requant zero err");

    for(int i=0;i<15;i++) CHECK(o->data_8[i]==0,"requant zero");

    printf("requantizeSIMD all-zero acc: ok(no NaN, scale=%f)\n", o->scale_factor);
    acc->data_32[3] = 2000000000; acc->data_32[4] = -2000000000;
    CHECK(requantizeSIMD(acc,o,127.0f)==0,"requant big err");
    CHECK(o->data_8[3]==127 && o->data_8[4]==-127,"requant big %d %d",o->data_8[3],o->data_8[4]);
    printf("requantizeSIMD +-2e9: %d %d\n", o->data_8[3], o->data_8[4]);
    deleteMatQ32(acc); deleteMatQ8(o);
}

static network_t make_net(const int *s, int n, float wv, float bv){
    network_t net; net.layer_quantity=n; net.max_layer_size=0;
    net.layers=_mm_malloc(sizeof(layer_t)*n,32);

    for(int i=0;i<n;i++){
        layer_t *l=&net.layers[i]; l->input_size=s[i]; l->output_size=s[i+1];

        if(s[i]>net.max_layer_size) net.max_layer_size=s[i];
        if(s[i+1]>net.max_layer_size) net.max_layer_size=s[i+1];

        l->weights=CreateMatrix(s[i],s[i+1]); l->biases=CreateMatrix(1,s[i+1]);

        for(int j=0;j<s[i]*s[i+1];j++) l->weights->data[j]=wv*(((j*7)%11)-5);
        for(int j=0;j<s[i+1];j++) l->biases->data[j]=bv;
    }
    return net;
}

static void test_fwd(const char *name, float in_v, float wv, float bv, int rows){
    int s[]={20,33,7};
    network_t net=make_net(s,2,wv,bv); q_network_t q;
    
    if(QuantizeNetwork(&net,&q)){ printf("%s: QuantizeNetwork FAILED\n",name); failures++; return; }
    matrix_t *in=CreateMatrix(rows,20);

    for(int i=0;i<rows*20;i++) in->data[i]=in_v*(((i*3)%7)-2);

    matrix_t *f=NULL,*o=QForwardPass(&q,in);
    if(rows==1) f=ForwardPass(&net,in);
    CHECK(o!=NULL,"%s: NULL",name);

    if(o){
        int nan=0; double num=0,den=0;
        for(int i=0;i<rows*7;i++){ if(!isfinite(o->data[i])) nan++; if(f){double d=f->data[i]-o->data[i];num+=d*d;den+=(double)f->data[i]*f->data[i];} }
        CHECK(nan==0,"%s: %d non finite",name,nan);
        CHECK(o->rows==rows && o->cols==7,"%s: shape %dx%d",name,o->rows,o->cols);
        printf("Forward %-26s rows=%d out[0]=%g ref=%g  %s rel.err=%.4f%%\n",name,rows,o->data[0], f?f->data[0]:0.0, f?"float-vs-quant":"(no float ref)", f?100*sqrt(num/(den>0?den:1)):0.0);
        if(f) CHECK(sqrt(num/(den>0?den:1))<0.03 || den==0,"%s: error too big",name);
    }
    DeleteMatrix(in); DeleteMatrix(o); if(f) DeleteMatrix(f);
    FreeNetPartial(&net,net.layer_quantity); _mm_free(net.layers); FreeQNetwork(&q);
}

int main(void){
    test_hand();
    test_uniform(127,127,1,"all 127 x 127");
    test_uniform(-127,127,3,"-127 x 127");
    test_uniform(-127,-127,129,"-127 x -127");
    test_uniform(127,127,4096,"max 127x127 K=4096");
    test_uniform(127,127,100000,"max K=100000 (<int32 max)");
    test_uniform(0,127,64,"zeros in A");
    test_uniform(127,0,64,"zeros in B");
    test_requant_zero();
    test_fwd("normal",1.0f,0.1f,0.1f,1);
    test_fwd("all-zero input",0.0f,0.1f,0.1f,1);
    test_fwd("all-zero input, bias 0",0.0f,0.1f,0.0f,1);
    test_fwd("all-zero weights",1.0f,0.0f,0.5f,1);
    test_fwd("huge input 1e15",1e15f,0.1f,0.1f,1);
    test_fwd("tiny input 1e-30",1e-30f,0.1f,0.1f,1);
    test_fwd("huge weights 1e15",1.0f,1e15f,0.1f,1);
    test_fwd("batch 5 rows",1.0f,0.1f,0.1f,5);
    //NaN / Inf inputs must be rejected, not crash

    { int s[]={20,33,7}; network_t net=make_net(s,2,0.1f,0.1f); q_network_t q; QuantizeNetwork(&net,&q);
      matrix_t *in=CreateMatrix(1,20); for(int i=0;i<20;i++) in->data[i]=1.0f; in->data[3]=INFINITY;
      matrix_t *o=QForwardPass(&q,in); printf("Inf input -> %s\n", o?"returned a matrix (BAD)":"NULL (rejected)"); CHECK(o==NULL,"inf accepted");
      DeleteMatrix(o); DeleteMatrix(in); FreeNetPartial(&net,2); _mm_free(net.layers); FreeQNetwork(&q); }
    printf("\n%s (%d failures)\n", failures?"SOME TESTS FAILED":"ALL EDGE TESTS PASSED", failures);
    return failures!=0;
}

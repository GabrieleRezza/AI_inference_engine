# NEURAL INFERENCE ENGINE

A simple yet complete project diving into the fundamentals of low-level programming for Artificial Intelligence. A full transition from a simple sistem to a complete engine with optimization, multi-threading and quantization.

# MAIN TARGET

This project is a personal learning journey behind the world of AI. 
My goal was to expand my portfolio, analizing the low-level programming of AI by build a full operational neural engine by hand with the help of Claude, which was guiding me through the project and helping me to correct my mistakes, while always keeping the distance and never writing code by itself.
This neural engine will probably be reused in a later project or rivisited later to adapt it to other uses.


## Project Structure

'''
AI_inference_engine/
├── tests/
│   ├── Benchmark_test.c          # Main file for benchmark tests
|   ├── Benchmark_test.exe
│   ├── matmul_test.c          # Main file to test correctness of matmul function
|   ├── matmul_test.exe
|   └── test.exe
├── export_C_data.py.py          # PY file used to import weights from PYTORCH
├── matrix.h            # matrix files contain declarations and functions to operate matrices
├── matrix.c    
├── network.h         # network files contain declarations and functions to 
├── network.c
└── weights_biases.bin       # HTML pages rendered via Flask routes
'''

## Input Weights-Biases Structure
Format of the input file for trained weights and biases loaded into the network

[magic_number]              (int32)
[num_layers]                (int32)
for every layer:
    [input_size]             (int32)
    [output_size]             (int32)
--- end of header ---
for every layer, in order:
    [weights]   input_size * output_size float, row-major
    [bias]   output_size float


## Prerequisites
- C compiler (MSYS2)
- Python 3.10+

## Usage


## Project timeline, problems and solutions
I started the project with the core structure and functions needed for the matrices: main struct, create, delete, initialize, clean and multiply functions. Since I already had a strong base in C programming I wrote a large majority of the functions alone and used Claude only to find typos and correct some mistakes.
After completing the matrix.c file and testing that everything worked by manually comparing the results with hand written tests, I confronted Claude to structure the next steps for the final days of the first week and the first half of the second one. 

Thanks to Claude's advices I understood the structure needed for the network file and started writing the main loading datas functions, while keeping track of the data structure itself, later reused to create an export_C_data file that translates PyTorch files into a usable file.bin for my project.
I also implemented some auxilary functions to simplify the readability of other functions under the tips given by Claude. During those days I discovered alone one of the optimizations for the Forward_pass, the ping pong buffers, and also discussed about cache blocking and loop reordering, understanding deeply their advantages in order to implement them in the best possible way.

After almost two weeks I completed the network files, checked their correctness by hand and with the starting of the third week around the corner I created a new file to keep track of the different times my CPU took to complete a matmul function based on the different sizes of the matrices, and therefore created a Benchmark table to document them all (all the results and considerations about benchmark tests can be found in the benchmark section).

During the third week I implemented loop reordering and cache blocking, testing different block sizes searching for the best optimization for my machine. As seen in the benchmark section, different sizes still give similar results with minor changes, but every single one of them, compared to tha native implementation is far more efficient.



# Benchmark tests
BenchmarkMatMul times based on matrix size and iterations

|    Size    |  Iterations  |  Naive(s)  | Blocking and Reordering  : --> | 16 | 32 | 64 || SIMD |     Speedup         | 
|------------|--------------|------------|--------------------------------|----|----|----||------|-----------------------|
|  Vect  x  256X256  | 1500 |  0.0004 s |  0.00035 s  |  0.00035 s  |  0.00035 s ||  0.000053 s  |  X 7.547
|  Vect  x  512x512  |  150 |  0.0022 s |  0.0014  s  |  0.00138 s  |  0.0014 s  ||   0.00025 s  |  X 8.80
| 256x256  x  256x256 | 100 |  0.096  s |  0.085/86 s |  0.086/87 s |  0.087 s   ||    0.012 s   |  X 8.00
| 512x512  x  512x512 | 10  |   1.13 s  |  0.69/70 s  |  0.69/70 s  |  0.697 s   ||    0.11 s    |  X 10.27
 
After experimenting with different sizes for cache blocking it can be observed a small increase in performance for Vect X Matrix multiplications: around 
12% and 36% respectively for 256x256 and 512x512. 
With MatrixXMatrix multiplications the impact is similar: 38% for 512x512 and 10% for 256x256.

It's clear that while using bigger matrices the impact of cache blocking becomes increasingly greater, while for smaller ones the changes are less notable and 
loop reordering is what is principally responsible for faster times.
Also, when vectors are involved, cache blocking is only partially exploited since it's less effective on the vector and it only affects the efficiency
when the matrix is larger.

In this project, since the kind of multiplication will always be vect X matrix, the benefit will always come proportionally from the size of the matrices.



|    Size    |  Iterations  |   Naive (s)  |  SIMD + Optimized Threading BLOCK 128 |     Speedup       | 
|  Vect  x  256X256  | 1500 |   0.0004 s   |             0.000082 s                |   X  4.878
|  Vect  x  512x512  |  150 |   0.0022 s   |             0.000140 s                |   X  15.71
| 256x256  x  256x256 | 100 |   0.096  s   |             0.0039   s                |   X  24.61 
| 512x512  x  512x512 | 10  |    1.13 s    |             0.0325   s                |   X  34.77


DEVO ORA CREARE FUNZIONE SEPARATA PER CASI CON RIGHE MINORI DI 512 IN MODO CHE SI USI SIMD E BASTA E RISPONDERE ALLE ULTIME DOMANDE DI CLAUDE
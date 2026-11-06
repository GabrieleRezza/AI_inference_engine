# NEURAL INFERENCE ENGINE

A simple yet complete project diving into the fundamentals of low-level programming for Artificial Intelligence. A full transition from a simple sistem to a complete engine with optimization, multi-threading and quantization.

# MAIN TARGET

This project is a personal learning journey behind the world of AI. 
My goal was to expand my portfolio, analizing the low-level programming of AI by build a full operational neural engine by hand with the help of Claude, which was guiding me through the project and helping me to correct my mistakes, while always keeping the distance and never writing code by itself.
This neural engine will probably be reused in a later project or rivisited later to adapt it to other uses.

> **Note on authorship:** the whole project has been written entirely by hand, with the only exception of the correction of the final steps of the quantization functions (the int8 matmul kernel `qMatmulAcc32`/`qMatmulSIMD`, `dequantizeAcc32` and the fixes to the other quantization functions) and the creation of the quantized forward pass (`QuantizeNetwork`, `QForwardPass`, `FreeQNetwork`) together with its test `tests/qmatmul_test.c`, which were implemented by Claude after a review of my code.
> The very last part of the project, the rewrite of `export_C_data.py` and the basic safety checks on the input data (both in the Python file and in `LoadNetworkHeader` inside `network.c`), was also implemented by Claude Code, as the final closing step of the initial project.

> **What this engine can and can not do:** it runs fully-connected networks (matrix multiplication + bias + ReLU between the layers, float or int8 quantized). It can hypothetically be used to test models like MNIST (digit recognition) and similar simple networks, but it can NOT run a real LLM, since embeddings, attention, normalizations and the other operations of modern models are not implemented.


## Project Structure

'''
AI_inference_engine/
├── tests/
│   ├── Benchmark_test.c          # Main file for benchmark tests
|   ├── Benchmark_test.exe
│   ├── matmul_test.c          # Main file to test correctness of matmul function
|   ├── matmul_test.exe
|   ├── qmatmul_test.c         # correctness of qMatmulAcc32, quantize/dequantize and QForwardPass vs ForwardPass
|   ├── qedge_test.c           # edge cases: hand computed values, zeros, +-127, huge/tiny values, Inf input
|   ├── Benchmark_quant.c      # float vs int8 benchmark (matmul and forward pass)
|   └── test.exe
├── export_C_data.py          # PY file that translates the weights and biases of a trained PyTorch network into weights_biases.bin
├── matrix.h            # matrix files contain declarations and functions to operate matrices
├── matrix.c    
├── network.h         # network files contain declarations and functions to 
├── network.c
├── quantization.h / quantization.c   # int8 quantization functions
└── weights_biases.bin       # binary file with the weights and biases, created by export_C_data.py
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
- C compiler (MSYS2) with support for AVX2 and OpenMP
- Python 3.10+ with `numpy`
- PyTorch (`torch`), needed only to export a `.pt` / `.pth` file. The `--demo` mode of the export file works without it

## Usage

**1. Create the weights file.** The export file reads a PyTorch network and writes `weights_biases.bin` in the format described above:

```
python export_C_data.py model.pt -o weights_biases.bin
python export_C_data.py --demo
```

`--demo` exports a small hardcoded network (2-3-2), useful to check that everything works without any model.

The network must be made only of `Linear` layers with a `ReLU` between them (and no ReLU after the last one), because this is exactly what the C engine computes. PyTorch saves the weights as (output, input), so the script transposes them to the (input, output) order used by the engine. At the end it reads the file again to check that the data is correct.

**2. Safety.** A `.pt` file is a "pickle", a Python format that can also contain code, and that code runs when the file is opened. For this reason the script by default loads only the tensors (`weights_only=True`) and refuses anything else. Save the model with `torch.save(model.state_dict(), "model.pt")` to use it directly. A whole saved model (`torch.save(model, ...)`) is accepted only with the `--trust` option, to be used only with files you created or trust. Never use it with files downloaded from unknown sources.

Basic checks are also done on the data: the script refuses files that are too big, networks with too many layers or too large sizes (max 256 layers, 2^20 neurons per layer, 2^28 parameters), layers that don't match each other and NaN/inf values. The same limits are checked by `LoadNetworkHeader` in the C code, so a corrupted `.bin` file is rejected before any memory is allocated.

**3. Run the engine.** Load the file with `LoadNetworkHeader` and `LoadNetworkWeights`, then use `ForwardPass` (float) or `QuantizeNetwork` + `QForwardPass` (int8). The tests and benchmarks in `tests/` are compiled like the one shown in the quantization benchmark section.

**Limits.** Hypothetically the engine can be used with MNIST and similar simple networks, but not with a real LLM (see the note at the beginning).


## Project timeline, problems and solutions
I started the project with the core structure and functions needed for the matrices: main struct, create, delete, initialize, clean and multiply functions. Since I already had a strong base in C programming I wrote a large majority of the functions alone and used Claude only to find typos and correct some mistakes.
After completing the matrix.c file and testing that everything worked by manually comparing the results with hand written tests, I confronted Claude to structure the next steps for the final days of the first week and the first half of the second one. 

Thanks to Claude's advices I understood the structure needed for the network file and started writing the main loading datas functions, while keeping track of the data structure itself, later reused to create an export_C_data file that translates PyTorch files into a usable file.bin for my project.
I also implemented some auxilary functions to simplify the readability of other functions under the tips given by Claude. During those days I discovered alone one of the optimizations for the Forward_pass, the ping pong buffers, and also discussed about cache blocking and loop reordering, understanding deeply their advantages in order to implement them in the best possible way.

After almost two weeks I completed the network files, checked their correctness by hand and with the starting of the third week around the corner I created a new file to keep track of the different times my CPU took to complete a matmul function based on the different sizes of the matrices, and therefore created a Benchmark table to document them all (all the results and considerations about benchmark tests can be found in the benchmark section).

During the third week I implemented loop reordering and cache blocking, testing different block sizes searching for the best optimization for my machine. As seen in the benchmark section, different sizes still give similar results with minor changes, but every single one of them, compared to the native implementation is far more efficient.

In the following weeks the main goal was to implement new optimization techniques: SIMD and threading. The challenge here was handling complex low-level algebra using functions which were completely new to me while keeping the main logic intact. My working procedure here consisted in planning the changes ahead with Claude's help, checking which functions to use for my project and consulting the official Intel Intrinsics Documentation online to check how every single function would affect data. Therefore I was able to always keep track of the single changes made by every function.

The real implementation was extremely complex, especially handling every edge case, AI helped me visualize those edge cases and guided me in the correct direction.

Testing the final functions after debugging everything showcased the incredible speedup brought by these implementations.

Finally the last step of this project was to implement quantization to optimize everything even further, so I naturalluy started by creating the core functions to handle this new type of data and matrices: new create, delete, clear functions and simple versions of quantize and dequantize.

During this week, months into the project, I realized that this step required far more work than expected, I implemented raw, naive functions for every quantization step after consulting Claude to create a working plan and using it as a guide and a debugging assistant for every new implementation. 
Having completed this initial version, the last thing to do was implement SIMD and threading in these new functions. Here is where Claude Code finally touched the codebase: after doing half the functions myself I decided to give Claude permission to finalize everything alone: completed qMatmulSIMD and created the last two functions, tested everything and made minor changes in the previous functions to clean them up of typos. The only big change that Claude suggested was to change the use of the SIMD maddubs function, getting read of a slow logic and presenting me with a new one. 

To test that everything still worked after Claude's work I run various new tests of quantized matmul and ForwardPass with successfull results.

As the closing step of the initial project I gave Claude Code the last two tasks. The first one was to completely restructure `export_C_data.py`, which was just a hardcoded example: now it reads a trained PyTorch network, checks that it is made only of Linear layers and ReLU (the only things my engine computes), transposes the weights, writes the binary file and reads it again to verify it. Claude also tested it with the demo network, whose result matches the one computed by hand.
The second task was security: I asked for simple basic checks to make sure that no dangerous data could be processed. Claude pointed out that the real risk is not the binary file but the PyTorch file itself, since it is a pickle that can execute code when opened, so the script now loads only tensors by default and the whole models need the explicit `--trust` option. It also added limits on layers and sizes, both in the Python file and in `LoadNetworkHeader`, which before this trusted the header completely and used its values to allocate memory.

While talking about this we also realized that, without embedding, attention and the other mechanisms, this engine can not run a real LLM. It stays a complete and working base for simple fully-connected networks (like MNIST), and the optimizations done (SIMD, threading, quantization) are the same ones used in the matrix multiplications of bigger models. The test with a real PyTorch file has not been done yet, because PyTorch was not installed on my machine.


# Benchmark tests
BenchmarkMatMul times based on matrix size and iterations

|    Size    |  Iterations  |  Naive(s)  | Blocking and Reordering  : --> | 16 | 32 | 64 || SIMD |     Speedup         | 
|------------|--------------|------------|--------------------------------|----|----|----||------|---------------------|
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
|  Vect  x  256X256  | 1500 |   0.0004 s   |             0.00008  s                |   X  5.00
|  Vect  x  512x512  |  150 |   0.0022 s   |             0.00014  s                |   X  15.71
| 256x256  x  256x256 | 100 |   0.096  s   |             0.0039   s                |   X  24.61 
| 512x512  x  512x512 | 10  |    1.13 s    |             0.0325   s                |   X  34.77

After implementing threading and optimizing it by creating two different functions (one for general matrices and one specifically designed for vectors), the results of the benchmark tests are better than expected. 
For bigger vectors we observe a doubling in speedup compared to just the SIMD function, while with matc=rices multiplications we are over the 3X mark compared to just the implementation of SIMD.
The only visible downside is with smaller vectors: here the threading actually slows down the process by 0.00003 s overall, due to the larger overhead in the instruction that unfortunately has a bigger impact than the improvement received by threading, therefore making it slightly worse. I still decided to keep this structure even if the previous version was better for performance because it was just a matter of copying the previous setup and for my projects aim I find this discovery more educationally relevant than a mere 0.0003 optimization of an edge case.

Tests using dynamic scheduling: little to no difference can be observed, the operating times are basically the same, just slightly worse for dynamic scheduling, unnoticeble in the the big picture. The simple static scheduling is perfect for this project since every matmul divides the multiplications evenly for every cicle of the for loop, making the overhead of the dynamic scheduling useless for what it actually brings as an optimization.


# Quantization benchmark (float vs int8)
Run with `gcc -O2 -mavx2 -mfma -fopenmp -o tests/Benchmark_quant.exe tests/Benchmark_quant.c matrix.c network.c quantization.c -lm`.
Functions used: `BenchmarkMatMul` (float), `BenchmarkQMatMul` (`qMatmulAcc32`) and `BenchmarkForward` (float `ForwardPass` / `QForwardPass`), all measuring wall-clock time.
Note: `BenchmarkMatMul` still uses `clock()`, whose resolution on Windows is ~1 ms, so the float times of the smallest sizes are coarse.

| Matmul size | float (s) | int8 (s) | speedup |
|---|---|---|---|
| 1x512 * 512x512 | 0.000120 | 0.000060 | 2.00x |
| 1x2048 * 2048x2048 | 0.001600 | 0.000340 | 4.71x |
| 64x512 * 512x512 | 0.001000 | 0.000580 | 1.72x |
| 256x256 * 256x256 | 0.001000 | 0.000400 | 2.50x |
| 512x512 * 512x512 | 0.004400 | 0.001720 | 2.56x |
| 1024x1024 * 1024x1024 | 0.029600 | 0.013200 | 2.24x |

| Forward pass (1 x n input) | float (s) | quantized (s) | speedup |
|---|---|---|---|
| 784-512-256-10 | 0.000420 | 0.000235 | 1.79x |
| 1024-2048-2048-1024 | 0.002660 | 0.001240 | 2.15x |
| 4096-4096-4096 | 0.014300 | 0.004200 | 3.40x |

Accuracy: relative error of the quantized forward pass vs the float one is ~0.1-1.1% on the test networks.


# Future developments
The project is considered complete as a first version, but it can be expanded in the future:

- **Test with a model trained by myself:** train a simple fully-connected network with PyTorch (for example on MNIST, with a structure like 784-128-10), save it with `torch.save(model.state_dict(), ...)`, export it with `export_C_data.py` and compare the accuracy and the speed of the float and of the int8 quantized version with the PyTorch one. This would be the first test of the whole pipeline on real data, since for now only the demo network and random test networks have been used.
- **Support for the `safetensors` format:** it is the format used by most of the models available online and it contains only tensors and no code, so it can be read without the pickle risks explained in the usage section.
- **Functions for LLMs:** to run a real language model the engine is missing embedding, attention, normalization layers (LayerNorm / RMSNorm), other activation functions (GELU, SiLU, softmax) and a different way to handle the input (tokens and a sequence instead of a single vector). The matrix multiplications, the SIMD, the threading and the quantization already done would be reused for these new parts.

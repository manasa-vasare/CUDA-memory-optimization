# Experiment 6: Memory Optimization in CUDA

[![CUDA](https://img.shields.io/badge/CUDA-13.5-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205060%20Ti-76B900?logo=nvidia)](https://www.nvidia.com/)
[![Compute Capability](https://img.shields.io/badge/Compute%20Capability-12.0-blue)](#-hardware--software-specifications)
[![Course](https://img.shields.io/badge/Course-Memory%20Systems%20%26%20Synchronization-darkgreen)](#)

> **Subject:** Memory Systems and Synchronization / Cloud & GPU Computing Laboratory  
> **Experiment No.:** 6  
> **Experiment Title:** Memory Optimization in CUDA — Coalesced vs. Uncoalesced Access, Naive Matrix Multiplication, and Execution-Time Comparison  
> **Target Platform:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), CUDA Toolkit 13.5, Windows 11 (64-bit)

---

## 🎯 Aim & Learning Outcomes

### Aim
To investigate how global-memory access patterns impact GPU performance by comparing an **uncoalesced** and a **coalesced** matrix transpose, evaluating a **naive matrix multiplication**, and analyzing the execution times, effective bandwidth (GB/s), and compute throughput (GFLOPS) measured on the GPU.

### Learning Outcomes
- Explain **memory coalescing** at the hardware warp level and why it is critical for maximizing global-memory bus bandwidth.
- Implement and launch 2D CUDA kernels utilizing `dim3` block and grid topologies.
- Employ **shared memory tiling** with conflict-free column padding (`[TILE][TILE + 1]`) to convert strided non-contiguous memory accesses into coalesced accesses.
- Benchmark asynchronous GPU kernel launches accurately using **CUDA events** (`cudaEventRecord`, `cudaEventElapsedTime`) with warm-up decoupling.
- Validate GPU kernel correctness against single-threaded host CPU references within numerical tolerance.

---

## 📌 Table of Contents

- [Aim & Learning Outcomes](#-aim--learning-outcomes)
- [Theoretical Foundations & Architecture](#-theoretical-foundations--architecture)
  - [1. Memory Coalescing & Warp Execution](#1-memory-coalescing--warp-execution)
  - [2. Matrix Transpose & Memory Access Patterns](#2-matrix-transpose--memory-access-patterns)
  - [3. Shared Memory Tiling & Bank Conflict Avoidance](#3-shared-memory-tiling--bank-conflict-avoidance)
  - [4. Naive Matrix Multiplication & The Memory Wall](#4-naive-matrix-multiplication--the-memory-wall)
  - [5. CUDA Event Timing Methodology](#5-cuda-event-timing-methodology)
- [Hardware & Software Specifications](#-hardware--software-specifications)
- [Project Structure & Implementation Steps](#-project-structure--implementation-steps)
- [Laboratory Benchmark Observations](#-laboratory-benchmark-observations)
  - [Observation Table (Section 6)](#observation-table-section-6)
  - [Multi-Run Stability Analysis](#multi-run-stability-analysis)
  - [Comparison with Reference Baseline (Tesla T4)](#comparison-with-reference-baseline-tesla-t4)
- [Performance Analysis & Discussion](#-performance-analysis--discussion)
- [Viva-Voce Questions & Comprehensive Answers](#-viva-voce-questions--comprehensive-answers)
- [Troubleshooting & Diagnostics](#-troubleshooting--diagnostics)
- [Compilation & Execution Guide](#-compilation--execution-guide)
- [Conclusion](#-conclusion)

---

## 🧠 Theoretical Foundations & Architecture

### 1. Memory Coalescing & Warp Execution

On NVIDIA GPUs, threads are grouped into physical execution units of **32 parallel threads** termed **warps**. When threads in a warp issue a load or store instruction to global DRAM:
- **Coalesced Access:** If the 32 threads access consecutive 32-bit (4-byte) or 64-bit memory addresses residing within aligned 32-byte or 128-byte cache line segments, the GPU memory controller merges the individual requests into as few as **1 to 2 wide bus transactions**. Memory bus utilization approaches the theoretical peak bandwidth.
- **Uncoalesced (Strided) Access:** If threads access scattered memory addresses or addresses separated by a large stride $N$ (e.g., column-wise access in row-major arrays), the hardware cannot merge the requests. The memory controller is forced to issue up to **32 distinct memory transactions**. Because each transaction transfers a minimum cache line chunk (32 or 128 bytes) while only 4 bytes are utilized per transaction, over **87.5% to 96.8% of the fetched memory bus bandwidth is wasted**.

| Characteristic | Coalesced Access | Uncoalesced (Strided) Access |
| :--- | :--- | :--- |
| **Address Pattern** | $\text{base} + i$ (consecutive) | $\text{base} + i \times N$ (stride $N$) |
| **DRAM Transactions / Warp** | Minimal (1 to 4 transactions) | Up to 32 transactions (serialized) |
| **Effective Bandwidth** | Close to hardware peak | Fractional bandwidth efficiency |

---

### 2. Matrix Transpose & Memory Access Patterns

Matrix transpose performs $\mathbf{B}[j][i] = \mathbf{A}[i][j]$. For an $N \times N$ matrix stored in row-major order, element $(r, c)$ resides at flat 1D index:
$$\text{Index}(r, c) = r \times N + c$$

#### Uncoalesced Kernel (`transposeNaive`)
```cpp
int x = blockIdx.x * TILE + threadIdx.x; // column index
int y = blockIdx.y * TILE + threadIdx.y; // row index
if (x < N && y < N)
    out[x * N + y] = in[y * N + x];
```
- **Read Operation (`in[y * N + x]`):** Consecutive threads in a warp have adjacent `threadIdx.x` values (same row $y$, consecutive columns $x$). Thus, global read addresses are contiguous $\rightarrow$ **Coalesced Read (Optimal)**.
- **Write Operation (`out[x * N + y]`):** Consecutive threads write to index $x \times N + y$. As `threadIdx.x` increases by 1, the write address jumps by $N$ elements (a massive stride of $N \times 4\text{ bytes}$). Threads hit 32 different cache lines $\rightarrow$ **Uncoalesced Write (Severe Bottleneck)**.

---

### 3. Shared Memory Tiling & Bank Conflict Avoidance

To eliminate uncoalesced global writes, the **Coalesced Transpose (`transposeCoalesced`)** stages data through ultra-fast, on-chip **Shared Memory**:

```cpp
__shared__ float tile[TILE][TILE + 1]; // +1 padding avoids bank conflicts

int x = blockIdx.x * TILE + threadIdx.x;
int y = blockIdx.y * TILE + threadIdx.y;

// 1. Coalesced Global Read into Shared Memory
if (x < N && y < N)
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];

__syncthreads(); // Barrier: wait for all 1024 threads in the 32x32 block

// 2. Transpose block coordinates
x = blockIdx.y * TILE + threadIdx.x;
y = blockIdx.x * TILE + threadIdx.y;

// 3. Coalesced Global Write from Transposed Shared Memory
if (x < N && y < N)
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];
```

#### The Role of Bank Padding (`TILE + 1`)
Shared memory is divided into **32 independent memory banks** (each 4 bytes wide). If multiple threads in a warp access different words mapped to the same bank simultaneously, a **shared-memory bank conflict** occurs, serializing access.
- Without padding (`tile[32][32]`): Thread $k$ reading column 0 accesses `tile[k][0]`. Since each row is 32 elements (a multiple of 32 banks), `tile[0][0], tile[1][0], \dots, tile[31][0]` all map to **Bank 0**, causing a **32-way bank conflict**.
- With $+1$ padding (`tile[32][33]`): Each row has 33 elements. Row $k$ starts at bank $(k \times 33) \pmod{32} = k \pmod{32}$. Reading down a column accesses banks $0, 1, 2, \dots, 31$, completely distributing accesses across all 32 banks and **eliminating bank conflicts**.

---

### 4. Naive Matrix Multiplication & The Memory Wall

For $N \times N$ matrices $\mathbf{C} = \mathbf{A} \times \mathbf{B}$, element $C[i][j]$ is defined as:
$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]$$

In the naive implementation (`matMulNaive`):
- Each thread independently computes one output element across an inner loop of $N$ iterations.
- Each iteration performs **2 global-memory reads** ($A[i][k]$ and $B[k][j]$) and **1 multiply-add (2 FLOPs)**.
- **Arithmetic Intensity:**
  $$\text{Arithmetic Intensity} = \frac{2 \times N^3\text{ FLOPs}}{2 \times N^3 \times 4\text{ bytes loaded}} = 0.25\text{ FLOP/byte} \quad (1\text{ FLOP per 4-byte load})$$
- Because the kernel lacks data reuse across threads via shared memory, the same elements of $A$ and $B$ are re-fetched from high-latency global DRAM thousands of times. The kernel is completely **memory-bound** and reaches only a tiny fraction of the GPU's peak compute throughput.

---

### 5. CUDA Event Timing Methodology

Because CUDA kernel launches are asynchronous (the CPU thread queues the launch on a stream and immediately returns), CPU wall-clock timers (`std::chrono`, `clock()`) do not measure kernel execution unless preceded by blocking synchronization, which introduces CPU driver scheduling overhead.
Instead, **CUDA Events** record GPU hardware timestamps directly on the GPU execution queue:
```cpp
cudaEvent_t s, e;
cudaEventCreate(&s);
cudaEventCreate(&e);

// Warm-up iteration (eliminates driver context initialization overhead)
k<<<g, b>>>(in, out, N);
cudaDeviceSynchronize();

cudaEventRecord(s);
for (int r = 0; r < reps; r++)
    k<<<g, b>>>(in, out, N);
cudaEventRecord(e);
cudaEventSynchronize(e);

float ms;
cudaEventElapsedTime(&ms, s, e);
float avg_time = ms / reps;
```

---

## 🖥 Hardware & Software Specifications

The laboratory benchmarks were compiled and evaluated natively on the target host workstation:

| Specification Category | Host Workstation Configuration |
| :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 5060 Ti** |
| **VRAM Capacity** | **16,311 MiB (16 GB GDDR6/GDDR7)** |
| **Compute Capability** | **12.0** (Next-Generation Blackwell Architecture) |
| **GPU Driver Version** | **621.24** (WDDM 3.2 Mode) |
| **CUDA Driver (UMD)** | **CUDA 13.5** |
| **CUDA Toolkit (`nvcc`)** | **Release 13.5, V13.5.37** (Build `cuda_13.5.r13.5/compiler.39125733_0`) |
| **Host C++ Compiler** | **Microsoft Visual C++ MSVC v143** (`cl.exe` v18.12.0) |
| **Operating System** | **Microsoft Windows 11 Pro 64-bit** |

---

## 📂 Project Structure & Implementation Steps

```text
CUDA-memory-optimization/
├── README.md               # Complete experimental report, analysis & viva answers
├── exp6.cu                 # Unified CUDA program (Steps 1–5 as in manual)
├── .gitignore              # Ignores build artifacts (.exe, .obj, bin/)
├── docs/
│   └── Experiment6_CUDA_Memory_Optimization_Manual.pdf  # Official Lab Manual
├── assets/
│   └── screenshots/        # Diagnostic and execution output logs
│       ├── setup_terminal.jpeg   # nvcc, nvidia-smi & compilation
│       ├── run1_benchmark.jpeg   # Benchmark Run 1 output
│       ├── run2_benchmark.jpeg   # Benchmark Run 2 output
│       └── run3_benchmark.jpeg   # Benchmark Run 3 output
└── src/
    ├── exp6.cu             # Complete unified CUDA program (Steps 1–5)
    └── phases/             # Step-by-step modular code breakdown
        ├── step1_headers_and_macro.cu   # Step 1: Headers & error check macro
        ├── step2_transpose_kernels.cu   # Step 2: Uncoalesced & Coalesced kernels
        ├── step3_matmul_kernel.cu       # Step 3: Naive matrix multiplication
        ├── step4_host_helpers.cu        # Step 4: CUDA event timing & verification
        └── step5_main.cu                # Step 5: Host memory, launches & reporting
```

---

## 📊 Laboratory Benchmark Observations

### Observation Table (Section 6)

The table below documents the measured performance for Matrix Transpose ($N = 4096$) and Naive Matrix Multiplication ($N = 1024$) on the **NVIDIA GeForce RTX 5060 Ti**:

| Kernel | Matrix Size ($N$) | Time (ms) | Speedup | Bandwidth (GB/s) / GFLOPS | Result (PASS/FAIL) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Transpose: Uncoalesced** | 4096 | **0.801 ms** *(mean)* | $1.0\times$ (baseline) | **167.6 GB/s** | **PASS** |
| **Transpose: Coalesced** | 4096 | **0.418 ms** *(mean)* | **$1.92\times$** | **321.5 GB/s** | **PASS** |
| **Naive Matrix Multiplication** | 1024 | **1.437 ms** *(mean)* | — | **1496.8 GFLOPS** | **PASS** |

* **Speedup of coalesced over uncoalesced transpose:** **$\mathbf{1.92\times}$**
* **GPU Model Used:** **NVIDIA GeForce RTX 5060 Ti**
* **Compute Capability:** **12.0**

---

### Multi-Run Stability Analysis

As mandated by **Section 4, Step 7** of the laboratory manual, the executable was run three consecutive times to confirm timing consistency and rule out thermal or clock throttling anomalies:

| Benchmark Parameter | Run 1 (`10:15:39 AM`) | Run 2 (`10:15:43 AM`) | Run 3 (`10:16:10 AM`) | Mean Stability Value |
| :--- | :---: | :---: | :---: | :---: |
| **Transpose Uncoalesced Time** | `0.797 ms` | `0.796 ms` | `0.809 ms` | **0.801 ms** |
| **Transpose Uncoalesced Bandwidth** | `168.4 GB/s` | `168.7 GB/s` | `165.8 GB/s` | **167.63 GB/s** |
| **Transpose Coalesced Time** | `0.417 ms` | `0.418 ms` | `0.418 ms` | **0.418 ms** |
| **Transpose Coalesced Bandwidth** | `322.1 GB/s` | `321.5 GB/s` | `320.8 GB/s` | **321.47 GB/s** |
| **Measured Transpose Speedup** | **$1.91\times$** | **$1.91\times$** | **$1.93\times$** | **$1.92\times$** |
| **Transpose Validation** | `PASS` | `PASS` | `PASS` | **All PASSED** |
| **Naive Matmul Time** | `1.399 ms` | `1.400 ms` | `1.511 ms` | **1.437 ms** |
| **Naive Matmul Throughput** | `1534.9 GFLOPS` | `1534.1 GFLOPS` | `1421.4 GFLOPS` | **1496.80 GFLOPS** |
| **Naive Matmul Validation** | `PASS` | `PASS` | `PASS` | **All PASSED** |

---

### Comparison with Reference Baseline (Tesla T4)

| Metric | Sample Reference (Tesla T4, CC 7.5) | Observed Workstation (RTX 5060 Ti, CC 12.0) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Uncoalesced Transpose Time** | `1.850 ms` | **0.797 ms** | **$2.32\times$ faster** |
| **Uncoalesced Bandwidth** | `145.1 GB/s` | **168.4 GB/s** | $+16.1\%$ higher |
| **Coalesced Transpose Time** | `0.720 ms` | **0.417 ms** | **$1.73\times$ faster** |
| **Coalesced Bandwidth** | `372.8 GB/s` | **322.1 GB/s** | RTX 5060 Ti memory bus saturation |
| **Transpose Speedup** | $2.57\times$ | **$1.92\times$** | Consistent with wider L2 cache absorption |
| **Naive Matmul Time** | `3.900 ms` | **1.399 ms** | **$2.79\times$ faster** |
| **Naive Matmul Throughput** | `550.6 GFLOPS` | **1534.9 GFLOPS** | **$2.79\times$ higher GFLOPS** |

---

## 🔬 Performance Analysis & Discussion

### 1. Why the Coalesced Transpose Wins
In the uncoalesced transpose kernel, every warp of 32 threads reads contiguous row data smoothly, but attempts to write across columns separated by $N$ elements ($4096 \times 4\text{ bytes} = 16,384\text{ bytes}$ apart). This breaks coalescing completely:
- Each write instruction requires up to **32 independent memory transactions**, each loading or writing an entire 32-byte cache line to alter just a single 4-byte float.
- In contrast, the coalesced kernel uses a $32 \times 32$ on-chip shared memory tile:
  1. Global memory is read contiguously row-wise $\rightarrow$ **Coalesced**.
  2. Data is transposed within on-chip SRAM (`tile[threadIdx.x][threadIdx.y]`).
  3. Swapped block coordinates ensure writes to global memory are contiguous row-wise $\rightarrow$ **Coalesced**.
- **Result:** Global bandwidth doubled from **$167.6\text{ GB/s}$** to **$321.5\text{ GB/s}$**, delivering a clean **$1.92\times$ speedup**.

### 2. Impact of Shared Memory Bank Padding (`TILE + 1`)
Shared memory on NVIDIA architectures comprises 32 distinct banks mapped cyclically every 32-bit word. 
- Without padding (`tile[32][32]`), reading down any column hits the exact same bank across all 32 threads, resulting in a **32-way bank conflict** that serializes access into 32 consecutive clock cycles.
- With $+1$ column padding (`tile[32][33]`), every consecutive row begins offset by 1 bank ($33 \pmod{32} = 1$). Reading column $c$ across rows $0 \dots 31$ touches banks $0, 1, 2, \dots, 31$, allowing simultaneous single-cycle access without conflict.

### 3. Naive Matrix Multiplication Bottleneck
The naive matrix multiplication achieved **$1534.9\text{ GFLOPS}$**. While impressive in isolation, this represents only a tiny fraction of the RTX 5060 Ti's theoretical peak FP32 compute capability.
- **Root Cause:** In the inner loop `sum += A[row * N + k] * B[k * N + col]`, every thread fetches $A[row * N + k]$ and $B[k * N + col]$ from global DRAM. Threads in the same block computing neighboring elements of $C$ repeatedly load the exact same rows of $A$ from DRAM without cache reuse.
- **Remedy:** The next logical optimization is **Tiled Matrix Multiplication**, which loads sub-blocks of $A$ and $B$ into shared memory once, enabling all threads in the block to reuse data $TILE$ times before fetching the next tile.

---

## ❓ Viva-Voce Questions & Comprehensive Answers

### Q1: What is a warp, and what does it mean for a memory access to be coalesced?
**Answer:**
- A **warp** is the basic unit of execution on an NVIDIA GPU, consisting of a group of **32 consecutive threads** executing in lockstep (SIMT: Single Instruction, Multiple Threads).
- A global memory access is **coalesced** when all 32 threads of a warp access neighboring, contiguous memory addresses within the same 32-byte or 128-byte memory segment. The memory controller merges these concurrent individual requests into a single or minimal number of wide memory transactions, maximizing memory bus efficiency.

---

### Q2: Which access in the uncoalesced transpose is strided: the read or the write? Why?
**Answer:**
- The **write access** is strided; the **read access is coalesced**.
- **Reasoning:** In row-major layout, address formula is $\text{index} = \text{row} \times N + \text{col}$. 
  - For the read `in[y * N + x]`, adjacent threads have adjacent $x$ coordinates (`threadIdx.x`), reading contiguous elements $x, x+1, x+2, \dots$ in row $y$.
  - For the write `out[x * N + y]`, thread $k$ writes to $x_k \times N + y$. As $x_k$ increases by 1 for each thread, the target address jumps by $N$ floats ($N \times 4\text{ bytes}$). The write stride is $N$, forcing 32 non-contiguous memory writes across distinct cache lines.

---

### Q3: Why is `__syncthreads()` required in the coalesced transpose?
**Answer:**
- `__syncthreads()` acts as a **block-wide barrier synchronization**.
- **Reasoning:** In the coalesced transpose, thread $(t_x, t_y)$ writes its loaded float into `tile[t_y][t_x]`. Later, the write step reads transposed data from `tile[t_x][t_y]`. Without `__syncthreads()`, fast threads might attempt to read `tile[t_x][t_y]` before the thread responsible for writing that cell has completed its load from global memory. This would cause a **Read-After-Write (RAW) data race** and corrupt the transposed matrix.

---

### Q4: What is a shared-memory bank conflict, and what does the `+1` padding do?
**Answer:**
- **Bank Conflict:** GPU shared memory is organized into **32 independent banks** (successive 32-bit words belong to successive banks $0 \dots 31$). When multiple threads in the same warp attempt to read or write different words belonging to the same memory bank simultaneously, the requests cannot be served concurrently and are serialized, reducing bandwidth by a factor proportional to the conflict degree (up to $32\times$).
- **Padding Function:** In a $32 \times 32$ 2D array, element `[r][c]` maps to bank $(r \times 32 + c) \pmod{32} = c \pmod{32}$. A column read across rows touches the exact same bank $c$. By declaring `tile[32][33]`, element `[r][c]` maps to bank $(r \times 33 + c) \pmod{32} = (r + c) \pmod{32}$. Reading column $c$ across rows $0 \dots 31$ accesses banks $0, 1, 2, \dots, 31$, completely distributing accesses and eliminating bank conflicts.

---

### Q5: Why do we use CUDA events instead of a CPU timer around the kernel launch?
**Answer:**
1. **Asynchronous Execution:** Kernel launches `kernel<<<grid, block>>>()` are non-blocking. The CPU issues the launch command to the driver queue and immediately advances to the next instruction. A CPU timer wrapped around the launch would measure only the CPU queuing overhead (microseconds), not actual GPU execution.
2. **Synchronous Flaw:** Forcing CPU synchronization with `cudaDeviceSynchronize()` around CPU timers measures kernel time polluted by operating system thread scheduling, PCIe driver signaling latency, and context-switching jitter.
3. **Hardware Precision:** CUDA events (`cudaEventRecord`) insert hardware timestamp markers directly into the GPU execution stream, measuring elapsed execution on the GPU with sub-microsecond accuracy.

---

### Q6: How many global-memory reads does one thread perform in the naive matmul for $N = 1024$?
**Answer:**
- In the naive matrix multiplication, each thread calculates one element $C[i][j]$ via a loop running $k = 0 \dots N-1$:
  $$\text{sum} += A[i \times N + k] \times B[k \times N + j]$$
- In each iteration, the thread performs **2 global-memory reads** (one float from $A$, one float from $B$).
- For $N = 1024$, one thread performs:
  $$\text{Total Global Reads per Thread} = 2 \times 1024 = \mathbf{2048\text{ reads}}$$
- Across all $1024 \times 1024$ threads, the kernel issues $2 \times 1024^3 \approx \mathbf{2.15 \times 10^9\text{ global loads}}$ from DRAM with zero reuse.

---

### Q7: How would shared-memory tiling reduce the global traffic of matrix multiplication?
**Answer:**
- In **Tiled Matrix Multiplication**, matrices $A$ and $B$ are partitioned into tiles of size $T \times T$ (e.g., $16 \times 16$ or $32 \times 32$).
- In each phase, the $T \times T$ threads of a thread block collectively perform **coalesced loads** of one tile of $A$ and one tile of $B$ into shared memory ($2 \times T^2$ float loads total for the entire block).
- After `__syncthreads()`, every thread in the block computes its partial dot product by reading the shared memory tile $T$ times.
- Data reuse factor is increased by $T$. Global DRAM traffic is reduced by a factor of $T$ (e.g., a **$16\times$ to $32\times$ reduction in global memory bandwidth demands**), lifting the kernel out of the memory-bound regime.

---

## 🛠 Troubleshooting & Diagnostics

| Problem | Root Cause | Verified Solution |
| :--- | :--- | :--- |
| `nvcc: command not found` | CUDA Toolkit not installed or not on system `PATH`. | Ensure nvcc is on PATH. On Windows use the **x64 Developer Command Prompt for Visual Studio**; on Colab use a GPU runtime. |
| `no CUDA-capable device detected` | Driver or GPU hardware uninitialized. | Confirm device status with `nvidia-smi`. Ensure display driver $\ge 525$ is installed. |
| `Result shows FAIL` | Incorrect 2D coordinate calculations or omitted block coordinate swap in coalesced write. | Check index expressions, especially swapped block coordinates in the coalesced kernel and bounds checks. |
| Negligible Speedup | Matrix size $N$ too small ($N \le 512$). | Small matrices fit in hardware L2 cache ($16\text{ MB}$ on RTX 5060 Ti). Use $N = 4096$ so memory bus traffic dominates. |

---

## 🚀 Compilation & Execution Guide

Follow the standard procedure from **Section 4 (Step 6: Compile & Step 7: Run and Record)** of the lab manual:

### Step 6: Compile

Open the **Developer Command Prompt for Visual Studio** (on Windows) or terminal (Linux / Colab) in the project directory:

```cmd
# Windows (run from a Developer Command Prompt)
nvcc -O2 -o exp6.exe exp6.cu

# Linux / Google Colab
nvcc -O2 -o exp6 exp6.cu
```

> **Note on Architecture Flags:** If needed, add `-arch=sm_XX` for your GPU (e.g., `-arch=sm_75` for a Colab T4, `-arch=sm_86` for RTX 30-series, or `-arch=sm_120` for RTX 50-series).

### Step 7: Run and Record

Run the executable directly from the terminal:

```cmd
# Windows: default sizes (transpose 4096, matmul 1024)
exp6.exe

# Windows: optional custom sizes (transpose N=2048, matmul N=512)
exp6.exe 2048 512

# Linux / Google Colab
./exp6
./exp6 2048 512
```

> **Manual Guideline:** Run the program at least **three times** and record the values in the observation table (Section 6). The first run after initializing the GPU device context can be slightly slower; use the repeated values.

---

## 🏁 Conclusion

Experiment 6 demonstrates that **memory access patterns are just as critical as raw arithmetic throughput** in high-performance GPU programming:
1. **Coalescing vs. Strided Access:** Transforming strided column writes into coalesced contiguous writes using on-chip shared memory yielded a **$1.92\times$ speedup** and nearly doubled effective memory bandwidth from **$167.6\text{ GB/s}$** to **$321.5\text{ GB/s}$** on an NVIDIA GeForce RTX 5060 Ti.
2. **Bank Conflicts:** Adding a $+1$ column padding (`tile[32][33]`) eliminated 32-way shared-memory bank conflicts when reading column data from the shared tile.
3. **Memory Wall in Naive Matmul:** The naive matrix multiplication highlighted the catastrophic impact of zero data reuse, demonstrating that staging data in shared memory is mandatory to escape memory-bound latency on modern GPUs.

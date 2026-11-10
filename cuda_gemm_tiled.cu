#include <cuda_runtime.h>
#include <cstdio>

#define TILE_SIZE 16

/*
HBM (A, B) 
  ↓ 每个线程搬一个元素
Shared Memory (sA, sB)
  ↓ 每个线程读一行×一列做点积
寄存器 (acc 累加)
  ↓ 最终写回
HBM (C)

*/

// Tiled GEMM: C = A * B
// __global__ 声明这是一个 CUDA kernel，由 CPU 启动、GPU 执行。参数 A、B 是输入矩阵，C 是输出矩阵，M/N/K 是矩阵维度：A 是 M×K，B 是 K×N，C 是 M×N。
__global__ void tiledGEMM(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    // 每个 block 负责 C 的一个 TILE_SIZE x TILE_SIZE 子块
    // 在 shared memory（片上 SRAM）里分配两个 tile 缓冲区。shared memory 是所有 block 内线程共享的，速度远快于 global memory（HBM），但容量有限。这里每个 block 各有一份 sA、sB。
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // 计算当前线程负责输出矩阵 C 中的哪个元素位置。blockIdx 决定在哪个 tile，threadIdx 决定在 tile 内的哪个位置。所以整个 grid 把 C 矩阵划分成 TILE_SIZE×TILE_SIZE 的子块，每个 block 算一个子块。
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 每个线程的局部累加器，用来存 C[row][col] 的部分和。初始化为 0。
    float acc = 0.0f;

    // 沿 K 维度分 tile，逐块从 HBM 搬运到 SRAM
    // 外层循环：沿 K 维度滑动。 (K + TILE_SIZE - 1) / TILE_SIZE 是向上取整，把 K 维度切成若干段，每段长度 TILE_SIZE。t 是第几个 K 段。
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // ---- 第1步: HBM -> SRAM ----
        // 每个线程负责加载 sA 和 sB 中的一个元素
        // 计算当前线程要从 global memory 搬运哪个位置的元素到 shared memory。aCol 是 A 矩阵的列索引，bRow 是 B 矩阵的行索引，都随 t 滑动。
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // 当前线程从 A 矩阵读一个元素，写入 shared memory 的 sA。row * K + aCol 是行优先布局下的线性索引。边界检查 row < M && aCol < K 防止越界，越界补 0。
        // 同理，从 B 矩阵读一个元素写入 sB。bRow * N + col 是 B 的线性索引。
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // 等待整个 block 把 tile 都搬进 SRAM
        // block 内同步屏障。 等所有线程都把自己的数据搬进 sA、sB 之后，才能开始计算。否则有的线程还没写完，别的线程就开始读，会读到脏数据。
        __syncthreads();

        // ---- 第2步: SRAM -> 寄存器 -> 计算 ----
        // 在当前 tile 内做点积。 当前线程取 sA 的一行（threadIdx.y 行）和 sB 的一列（threadIdx.x 列），逐元素相乘累加。这本质上就是算 C[row][col] 在当前 K 段上的部分贡献。
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // 所有线程用完当前 tile 后，才能加载下一个 tile
        // 第二次同步。 等所有线程都用完当前 sA、sB 之后，才能进入下一轮 t 循环，覆盖 sA、sB 搬下一批数据。如果不加这个，前一个线程还在用 sA，后一个线程就把 sA 覆盖了，结果会出错。
        __syncthreads();
    }

    // ---- 第3步: 寄存器 -> SRAM(隐式) -> HBM ----
    // 把最终结果写回 global memory。边界检查防止越界写（比如 grid 比矩阵大时，多余的线程不写）。
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int main() {
    const int M = 1024, K = 1024, N = 1024;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // Host 分配
    float *hA = new float[M * K], *hB = new float[K * N], *hC = new float[M * N];
    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = 1.0f;

    // Device 分配 (HBM)
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    // Host -> Device (PCIe -> HBM)
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    tiledGEMM<<<grid, block>>>(dA, dB, dC, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);

    // Device -> Host (HBM -> PCIe)
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);
    printf("C[0] = %.1f (expect %d.0)\n", hC[0], K);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}

/*
① A[row*K+aCol]   —— global load（读全局内存）
       │
       │  ★ 硬件自动走完整缓存层次，代码里看不见
       ▼
   ┌──────────────────────────────────────────┐
   │  L1（SM 内，~30 cycle）  ← 先查这里      │  命中? 直接用
   │      ↓ miss                              │
   │  L2（芯片级，~200 cycle） ← 再查这里     │  命中? 从这拿
   │      ↓ miss                              │
   │  HBM（~500 cycle，936 GB/s） ← 最后才去  │
   └──────────────────────────────────────────┘
       │  数据最终进入【寄存器】GPU 是 load-store 架构——没有"内存到内存"的直接搬运指令，所有数据移动都必须用手（寄存器）过一遍。
       ▼
② sA[ty][tx] = ...  —— shared store（写共享内存）
       │
       │ ★ 这是独立地址空间，直接写到片上 SRAM
       │   不经过 L1/L2，也不占 HBM 带宽
       ▼
   片上 SRAM 里的一小块（就是你 __shared__ 申请的那块）
*/
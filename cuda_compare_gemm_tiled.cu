// cuda_compare_gemm_tiled.cu —— 单文件：naive GEMM vs tiled GEMM
//   同输入 / 同 grid / 同 block / 同一份校验，打印输入输出窗口 + 性能对比
// 编译：
//   nvcc -O3 -arch=sm_86 -o cuda_compare_gemm_tiled cuda_compare_gemm_tiled.cu
// 运行：
//   ./cuda_compare_gemm_tiled          # 默认 N=1024
//   ./cuda_compare_gemm_tiled 4        # ★ 小尺寸：A/B 输入 + 两种 C 输出全量打印
//   ./cuda_compare_gemm_tiled 513      # 非对齐：验证边界 mask
//   ./cuda_compare_gemm_tiled 2048     # 大尺寸看加速比
//   ./cuda_compare_gemm_tiled 32 8     # 第二个参数 = 打印窗口边长
//
// 方阵 A/B/C 都是 N×N，行主序 fp32。
// 数据填 1/16 的整数倍 → fp32 可精确表示，三者能逐 bit 对齐（max_abs=0）。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define TILE_SIZE 16

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ==================== kernel 1：普通 GEMM（原版，一行未改）====================
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ==================== kernel 2：shared memory 分块 GEMM（原逻辑未改）====================
__global__ void tiledGEMM(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
// ============================================================================

// ---------- host 侧打印：一个 W×W 窗口 ----------
static void print_window(const char* name, const float* M,
                         int N, int W, int r0, int c0) {
    printf("%s  行[%d..%d] 列[%d..%d]:\n",
           name, r0, r0 + W - 1, c0, c0 + W - 1);
    for (int r = r0; r < r0 + W; ++r) {
        printf("   ");
        for (int c = c0; c < c0 + W; ++c) {
            float v = M[(size_t)r * N + c];
            if (std::isnan(v)) printf("      nan");
            else               printf("%9.4f", v);
        }
        printf("\n");
    }
}

// ---------- 同一窗口的 double 真值（O(W²·N)）----------
static void print_golden_window(const char* name, const float* A, const float* B,
                                int N, int W, int r0, int c0) {
    printf("%s (CPU double 真值)  行[%d..%d] 列[%d..%d]:\n",
           name, r0, r0 + W - 1, c0, c0 + W - 1);
    for (int r = r0; r < r0 + W; ++r) {
        printf("   ");
        for (int c = c0; c < c0 + W; ++c) {
            double ref = 0.0;
            for (int k = 0; k < N; ++k)
                ref += (double)A[(size_t)r * N + k] * (double)B[(size_t)k * N + c];
            printf("%9.4f", ref);
        }
        printf("\n");
    }
}

// ---------- 计时：warmup + reps 次取平均 ----------
template <typename Launch>
static float bench_ms(Launch launch, int reps) {
    launch();                                   // warmup，吃掉首次上下文/启动开销
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) { fprintf(stderr, "N 必须为正整数\n"); return 1; }
    const int W_req = (argc > 2) ? atoi(argv[2]) : (N <= 8 ? N : 4);
    const int W = std::max(1, std::min(W_req, N));

    const int M = N, K = N;
    const int GRID = (N + TILE_SIZE - 1) / TILE_SIZE;
    const size_t bytes = (size_t)N * N * sizeof(float);
    const int reps = (N <= 512) ? 10 : (N <= 2048 ? 5 : 3);

    printf("============================================================\n");
    printf("naive GEMM  vs  tiled GEMM(TILE=%d)  —— 同输入 / 同 grid / 同 block\n",
           TILE_SIZE);
    printf("M=N=K=%d | block=%dx%d | grid=%dx%d | reps=%d | 打印窗口=%dx%d\n",
           N, TILE_SIZE, TILE_SIZE, GRID, GRID, reps, W, W);
    printf("============================================================\n\n");

    // ---------- 同一份输入 ----------
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_naive = (float*)malloc(bytes);
    float *hC_tiled = (float*)malloc(bytes);
    if (!hA || !hB || !hC_naive || !hC_tiled) {
        fprintf(stderr, "host malloc 失败\n"); return 1;
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            hA[(size_t)i * N + j] = (float)(((i + 2 * j) % 19) - 9) * 0.0625f;
            hB[(size_t)i * N + j] = (float)(((3 * i + j) % 23) - 11) * 0.0625f;
            hC_naive[(size_t)i * N + j] = NAN;   // canary：检测漏写
            hC_tiled[(size_t)i * N + j] = NAN;
        }

    printf("【输入 A】\n");   print_window("A", hA, N, W, 0, 0);
    printf("\n【输入 B】\n"); print_window("B", hB, N, W, 0, 0);
    printf("\n");

    // ---------- device：A/B 只有一份，两个 C 各自一份 ----------
    float *dA, *dB, *dC_naive, *dC_tiled;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC_naive, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC_tiled, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_naive, hC_naive, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_tiled, hC_tiled, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(GRID, GRID);

    // ★ 计时期间 kernel 内不做任何 printf（device printf 会严重拖慢两者，比出来的数没意义）
    float ms_naive = bench_ms([&]() {
        matmul_naive<<<grid, block>>>(dA, dB, dC_naive, N);
    }, reps);
    float ms_tiled = bench_ms([&]() {
        tiledGEMM<<<grid, block>>>(dA, dB, dC_tiled, M, N, K);
    }, reps);

    CUDA_CHECK(cudaMemcpy(hC_naive, dC_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_tiled, dC_tiled, bytes, cudaMemcpyDeviceToHost));

    // ---------- 打印两种输出 + 真值 ----------
    printf("【输出 C —— naive · 左上】\n");
    print_window("C_naive", hC_naive, N, W, 0, 0);
    printf("\n【输出 C —— tiled · 左上】\n");
    print_window("C_tiled", hC_tiled, N, W, 0, 0);
    printf("\n【CPU 真值 —— 左上】\n");
    print_golden_window("golden", hA, hB, N, W, 0, 0);

    if (N > W) {
        const int r0 = N - W, c0 = N - W;
        printf("\n【输出 C —— naive · 右下（边界 mask）】\n");
        print_window("C_naive", hC_naive, N, W, r0, c0);
        printf("\n【输出 C —— tiled · 右下（边界 mask）】\n");
        print_window("C_tiled", hC_tiled, N, W, r0, c0);
        printf("\n【CPU 真值 —— 右下】\n");
        print_golden_window("golden", hA, hB, N, W, r0, c0);
    }

    // ---------- 校验 ----------
    // ① 全量：naive 与 tiled 应逐 bit 一致（两者 K 轴累加顺序相同，补 0 不改变 fp32 值）
    long long diff_nt = 0, nan_naive = 0, nan_tiled = 0;
    double max_nt = 0.0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        if (std::isnan(hC_naive[i])) nan_naive++;
        if (std::isnan(hC_tiled[i])) nan_tiled++;
        if (hC_naive[i] != hC_tiled[i]) {
            diff_nt++;
            if (!std::isnan(hC_naive[i]) && !std::isnan(hC_tiled[i]))
                max_nt = std::max(max_nt,
                    (double)fabsf(hC_naive[i] - hC_tiled[i]));
        }
    }

    // ② 抽样对 double 真值（全量 golden 是 O(N³)，太大时不可行）
    const int SAMPLES = 4096;
    unsigned rng = 12345u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    int bad_gold = 0;
    double max_gold = 0.0;
    for (int s = 0; s < SAMPLES; ++s) {
        int i = (int)(next() % (unsigned)N);
        int j = (int)(next() % (unsigned)N);
        double ref = 0.0;
        for (int k = 0; k < N; ++k)
            ref += (double)hA[(size_t)i * N + k] * (double)hB[(size_t)k * N + j];
        double d1 = fabs((double)hC_naive[(size_t)i * N + j] - ref);
        double d2 = fabs((double)hC_tiled[(size_t)i * N + j] - ref);
        double tol = 1e-3 * std::max(1.0, fabs(ref));
        if (d1 > tol || d2 > tol) bad_gold++;
        max_gold = std::max(max_gold, std::max(d1, d2));
    }

    printf("\n------------------------------------------------------------\n");
    printf("校验① naive vs tiled : diff=%lld 个, max_abs=%.3e %s\n",
           diff_nt, max_nt,
           (diff_nt == 0 && nan_naive == 0 && nan_tiled == 0) ? "✅" : "❌");
    printf("校验② 抽样 vs double : bad=%d/%d, max_abs=%.3e %s\n",
           bad_gold, SAMPLES, max_gold, bad_gold ? "❌" : "✅");
    printf("NaN 残留(漏写)      : naive=%lld tiled=%lld %s\n",
           nan_naive, nan_tiled, (nan_naive || nan_tiled) ? "❌" : "✅");

    // ---------- 性能对比 ----------
    auto gflops = [&](float ms) { return 2.0 * M * N * K / (ms * 1e6); };
    printf("\n========== 性能对比（同输入 / 同 grid / 同 block / %d 次平均）==========\n",
           reps);
    printf("%-14s %12s %14s\n", "kernel", "time (ms)", "GFLOPS");
    printf("%-14s %12.3f %14.2f\n", "matmul_naive", ms_naive, gflops(ms_naive));
    printf("%-14s %12.3f %14.2f\n", "tiledGEMM(16)", ms_tiled, gflops(ms_tiled));
    printf("%-14s %12.2f x\n", "加速比", ms_naive / ms_tiled);
    printf("\n理论 global 读量: naive = 8·N³ B, tiled ≈ 8·N³/%d B (降 %d 倍)\n",
           TILE_SIZE, TILE_SIZE);
    printf("注意: L1/L2 会缓存 naive 反复读的 A 行/B 列，所以实测加速比通常只有几倍，\n"
           "      不会等于理论 %d 倍；这正是『算术强度 + 缓存层次』的考点。\n",
           TILE_SIZE);
    printf("============================================================\n");

    bool ok = (diff_nt == 0 && bad_gold == 0 && nan_naive == 0 && nan_tiled == 0);
    printf("总体: %s\n", ok ? "✅ 两个 kernel 结果一致且正确" : "❌ 存在不一致，别看性能");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_naive)); CUDA_CHECK(cudaFree(dC_tiled));
    free(hA); free(hB); free(hC_naive); free(hC_tiled);
    return ok ? 0 : 1;
}
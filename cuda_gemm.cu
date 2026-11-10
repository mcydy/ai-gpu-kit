// cuda_gemm.cu —— 单文件、单 kernel 的 CUDA GEMM（带输入/输出打印）
//
// 编译（3090 / SM86）：
//   nvcc -O3 -arch=sm_86 -o cuda_gemm cuda_gemm.cu
// 运行：
//   ./cuda_gemm          # N=1024，打印左上 4x4 窗口
//   ./cuda_gemm 4        # ★ 学习用：N=4，A/B/C 全量打印 + C[0][0] 点积展开
//   ./cuda_gemm 513      # 非对齐尺寸，右下角窗口专门验证 row<N && col<N 边界 mask
//   ./cuda_gemm 513 6    # 指定打印窗口为 6x6
//   ./cuda_gemm 2048
//
// C = A * B，行主序，N×N 方阵，fp32。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ==================== 唯一的 kernel（原版，一行未改） ====================
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 当前线程负责的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 当前线程负责的列
    if (row < N && col < N) {                          // 非对齐 N 的边界 mask
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
// ============================================================================

// ---------- 打印一个 W×W 窗口；(r0,c0) 是左上角起点 ----------
static void print_window(const char* name, const float* M,
                         int N, int W, int r0, int c0) {
    printf("%s  行[%d..%d] 列[%d..%d]:\n",
           name, r0, r0 + W - 1, c0, c0 + W - 1);
    for (int r = r0; r < r0 + W; r++) {
        printf("   ");
        for (int c = c0; c < c0 + W; c++) {
            float v = M[(size_t)r * N + c];
            if (std::isnan(v)) printf("      nan");   // canary：没被 kernel 写入
            else               printf("%9.4f", v);
        }
        printf("\n");
    }
}

// ---------- 打印同一窗口的 CPU double 真值（O(W²·N)，很快）----------
static void print_golden_window(const char* name, const float* A, const float* B,
                                int N, int W, int r0, int c0) {
    printf("%s (CPU double 真值)  行[%d..%d] 列[%d..%d]:\n",
           name, r0, r0 + W - 1, c0, c0 + W - 1);
    for (int r = r0; r < r0 + W; r++) {
        printf("   ");
        for (int c = c0; c < c0 + W; c++) {
            double ref = 0.0;
            for (int k = 0; k < N; k++)
                ref += (double)A[(size_t)r * N + k] * (double)B[(size_t)k * N + c];
            printf("%9.4f", ref);
        }
        printf("\n");
    }
}

// ---------- 把 C[0][0] 那一行 kernel 代码展开给人看 ----------
static void print_dot_expand(const float* A, const float* B, int N, float c00) {
    printf("C[0][0] 点积展开  (kernel 里的 sum += A[0][k] * B[k][0]):\n");
    const int show = (N <= 12) ? N : 4;        // 大 N 只打头尾
    float acc = 0.0f;
    for (int k = 0; k < N; k++) {
        float p = A[(size_t)0 * N + k] * B[(size_t)k * N + 0];
        acc += p;                              // fp32 累加，和 kernel 完全一致
        if (k < show || k >= N - 2)
            printf("   k=%d  A[0][%d]=%8.4f  *  B[%d][0]=%8.4f  = %10.6f   sum=%10.6f\n",
                   k, k, A[(size_t)0 * N + k], k, B[(size_t)k * N + 0], p, acc);
        else if (k == show)
            printf("   ... 省略中间 %d 项 ...\n", N - 2 - show);
    }
    printf("   => C[0][0] = %.6f  (GPU 回读值 = %.6f)\n", acc, c00);
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) { fprintf(stderr, "N 必须为正整数\n"); return 1; }
    const int W_req = (argc > 2) ? atoi(argv[2]) : (N <= 8 ? N : 4);
    const int W     = std::max(1, std::min(W_req, N));   // 打印窗口边长

    const int TILE = 16;                                  // blockDim = 16x16 = 256 线程
    const int GRID = (N + TILE - 1) / TILE;              // cdiv，N 非 16 倍数时多起一块
    const size_t bytes = (size_t)N * N * sizeof(float);

    printf("============================================================\n");
    printf("N=%d | block=%dx%d | grid=%dx%d | %.1f MiB/矩阵 | 打印窗口=%dx%d\n",
           N, TILE, TILE, GRID, GRID, bytes / 1048576.0, W, W);
    printf("线程: 启动 %lld, 有效 %lld (%.1f%%) —— 其余被 row<N && col<N 挡住\n",
           (long long)GRID * GRID * TILE * TILE, (long long)N * N,
           100.0 * (double)N * N / ((double)GRID * GRID * TILE * TILE));
    printf("============================================================\n\n");

    // ---------- host 数据：1/16 的整数倍，fp32 可精确表示 ----------
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc 失败\n"); return 1; }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            hA[(size_t)i * N + j] = (float)(((i + 2 * j) % 19) - 9) * 0.0625f;
            hB[(size_t)i * N + j] = (float)(((3 * i + j) % 23) - 11) * 0.0625f;
            hC[(size_t)i * N + j] = NAN;                 // canary：检测漏写
        }

    // ---------- 打印输入 ----------
    printf("【输入 A】\n");
    print_window("A", hA, N, W, 0, 0);
    printf("\n【输入 B】\n");
    print_window("B", hB, N, W, 0, 0);
    printf("\n");

    // ---------- device ----------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, bytes, cudaMemcpyHostToDevice));  // 把 NaN 带下去

    dim3 block(TILE, TILE);
    dim3 grid(GRID, GRID);

    // warmup：吃掉首次上下文/启动开销
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 计时 ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ---------- 打印输出：左上 + 右下（右下专门看边界 mask）----------
    printf("【输出 C —— 左上】\n");
    print_window("C", hC, N, W, 0, 0);
    printf("\n【CPU 真值 —— 左上】\n");
    print_golden_window("golden", hA, hB, N, W, 0, 0);

    if (N > W) {
        const int r0 = N - W, c0 = N - W;
        printf("\n【输出 C —— 右下（边界 mask 验证）】\n");
        print_window("C", hC, N, W, r0, c0);
        printf("\n【CPU 真值 —— 右下】\n");
        print_golden_window("golden", hA, hB, N, W, r0, c0);
    }

    printf("\n");
    print_dot_expand(hA, hB, N, hC[0]);

    // ---------- canary：所有合法位置都必须被写过 ----------
    long long nan_cnt = 0;
    for (size_t i = 0; i < (size_t)N * N; i++)
        if (std::isnan(hC[i])) nan_cnt++;
    printf("\nNaN 残留(未被写入的元素) = %lld %s\n",
           nan_cnt, nan_cnt ? "❌" : "✅");

    // ---------- 抽样对 double 真值校验（全量 O(N³) 太慢）----------
    const int SAMPLES = 4096;
    unsigned rng = 12345u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };

    int bad = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (int s = 0; s < SAMPLES; s++) {
        int i = (int)(next() % (unsigned)N);
        int j = (int)(next() % (unsigned)N);
        double ref = 0.0;
        for (int k = 0; k < N; k++)
            ref += (double)hA[(size_t)i * N + k] * (double)hB[(size_t)k * N + j];
        double diff = fabs((double)hC[(size_t)i * N + j] - ref);
        double tol  = 1e-3 * std::max(1.0, fabs(ref));
        if (diff > tol) bad++;
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / std::max(1.0, fabs(ref)));
    }

    double gflops = 2.0 * N * N * N / (ms * 1e6);
    printf("抽样校验: %d 样本 | max_abs=%.3e max_rel=%.3e | bad=%d %s\n",
           SAMPLES, max_abs, max_rel, bad, bad ? "❌" : "✅");
    printf("耗时: %.2f ms | %.2f GFLOPS\n", ms, gflops);
    if (bad == 0 && nan_cnt == 0)
        printf("提示: naive 每个 C 元素读 2N 个 float，纯带宽瓶颈；"
               "加速要靠 shared memory 分块 / Tensor Core。\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return (bad || nan_cnt) ? 1 : 0;
}
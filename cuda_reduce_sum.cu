// cuda_reduce_sum.cu —— 单文件：block 内树形归约 + main + 输入/输出打印 + 校验
//
// 编译（3090 / SM86）：
//   nvcc -O3 -arch=sm_86 -o cuda_reduce_sum cuda_reduce_sum.cu
// 运行：
//   ./cuda_reduce_sum          # 默认 n=600（3 个 block，最后一个补 0 最直观）
//   ./cuda_reduce_sum 8        # 小尺寸：输入全量打印
//   ./cuda_reduce_sum 257      # 非 256 倍数：看补 0 线程
//   ./cuda_reduce_sum 4096     # 看带宽
//
// ★ output 数组长度 = GRID（每个 block 一个局部和），不是 n。
//   grid 用 cdiv(n, BLOCK)，blockDim 必须是 2 的幂（s >>= 1 的要求）。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define BLOCK 256   // 2 的幂，树形归约要求

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                       \
    } while (0)

// ==================== 唯一的 kernel（你给的原版，一行未改）====================
// 并行归约求和 kernel：每个 block 负责 input 中一段长度为 blockDim.x 的子数组，
// 在 Shared Memory 里做树形两两累加，最后由 0 号线程输出该 block 的局部和
__global__ void reduce_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[]; // Shared Memory 数组，大小由 kernel 启动时的第三个配置参数指定
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;  // 越界补 0：加进 sum 不改变结果
    __syncthreads();

    // 树形归约：s = 128,64,32,...,1，每轮把右半折到左半，共 log2(BLOCK) 轮
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();   // ★ 在 if 外面：block 内所有线程都必须到达，不能放进 if
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];   // ★ 写的是 GRID 个局部和
}
// ============================================================================

// ---------- 打印 1D 数组窗口（带下标；idx>=valid_n 显示 ---，且不访问内存）----------
static void print_vec(const char* name, const float* v,
                      int start, int len, int valid_n) {
    printf("%s  下标[%d..%d]:\n", name, start, start + len - 1);
    for (int s = start; s < start + len; s += 8) {
        int e = std::min(s + 8, start + len);
        printf("   idx:");
        for (int i = s; i < e; ++i) printf("%8d", i);
        printf("\n   val:");
        for (int i = s; i < e; ++i) {
            if (i >= valid_n)          printf("     ---");   // 虚拟线程（无对应元素）
            else if (std::isnan(v[i])) printf("     nan");   // 没被 kernel 写入
            else                       printf("%9.5f", v[i]);
        }
        printf("\n");
    }
}

// ---------- 打印 output[GRID]：每个 block 的局部和 + 该段的 CPU golden ----------
static void print_partials(const float* out, const float* in,
                           int n, int GRID) {
    printf("【输出 output[GRID] —— 每个 block 的局部和】\n");
    printf("   block   全局下标范围      有效元素   局部和        CPU golden   差\n");
    for (int b = 0; b < GRID; ++b) {
        int lo = b * BLOCK;
        int hi = std::min(n, lo + BLOCK);
        double g = 0.0;
        for (int i = lo; i < hi; ++i) g += (double)in[i];  // 只算有效元素，等价于"越界补 0"
        printf("   %5d   [%6d..%6d)   %6d   %11.5f   %11.5f   %.1e\n",
               b, lo, lo + BLOCK, hi - lo, out[b], g,
               fabs((double)out[b] - g));
    }
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 600;
    if (n <= 0) { fprintf(stderr, "n 必须为正整数\n"); return 1; }

    const int GRID = (n + BLOCK - 1) / BLOCK;   // cdiv
    const int padded = GRID * BLOCK;            // 启动线程总数
    const int guard = padded - n;               // ★ 越界、补 0 的虚拟线程数
    const size_t bytesIn  = (size_t)n * sizeof(float);
    const size_t bytesOut = (size_t)GRID * sizeof(float);   // ★★ 不是 n！
    const size_t shmem = (size_t)BLOCK * sizeof(float);     // extern __shared__
    const int W = std::min(n, 8);               // 输入打印窗口
    const int reps = 20;

    printf("============================================================\n");
    printf("reduce_sum (block 内树形归约) | n=%d | block=%d | grid=%d\n", n, BLOCK, GRID);
    printf("启动线程=%d, 有效=%d, 补 0=%d | shared=%.1f KB | output 长度=%d\n",
           padded, n, guard, shmem / 1024.0, GRID);
    printf("============================================================\n\n");

    // ---------- 输入：1/16 整数倍，fp32 精确，求和逐 bit 可对齐 ----------
    float *hIn  = (float*)malloc(bytesIn);
    float *hOut = (float*)malloc(bytesOut);
    if (!hIn || !hOut) { fprintf(stderr, "host malloc 失败\n"); return 1; }
    for (int i = 0; i < n; ++i)
        hIn[i] = (float)(((i % 19) - 9)) * 0.0625f;
    for (int b = 0; b < GRID; ++b)
        hOut[b] = NAN;                          // canary：哪个 block 没写就露出来

    // ---------- 打印输入：头部 + 跨 n 的尾部（--- 即虚拟线程补 0 的位置）----------
    print_vec("【输入 input · 头部】", hIn, 0, W, n);
    printf("\n");
    if (n > W) {
        int t0 = n - W;
        int tlen = W + std::min(guard, 8);
        print_vec("【输入 input · 尾部 + 虚拟线程】", hIn, t0, tlen, n);
        printf("   (idx >= %d 显示 --- ：这些线程走 sdata[tid]=0.0f，不影响局部和)\n", n);
    }
    printf("\n");

    // ---------- device ----------
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc((void**)&dIn,  bytesIn));
    CUDA_CHECK(cudaMalloc((void**)&dOut, bytesOut));
    CUDA_CHECK(cudaMemcpy(dIn,  hIn,  bytesIn,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOut, hOut, bytesOut, cudaMemcpyHostToDevice));  // NAN 带下去

    dim3 grid(GRID);
    dim3 block(BLOCK);

    // warmup，吃掉首次 CUDA 上下文/启动开销
    reduce_sum<<<grid, block, shmem>>>(dIn, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i)
        reduce_sum<<<grid, block, shmem>>>(dIn, dOut, n);  // ★ 计时期间 kernel 内不 printf
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= reps;

    CUDA_CHECK(cudaMemcpy(hOut, dOut, bytesOut, cudaMemcpyDeviceToHost));

    // ---------- 打印输出 ----------
    print_partials(hOut, hIn, n, GRID);

    // ★ 多 block 结果再 reduce：host 端把 GRID 个局部和累加成最终标量
    double sum_partial = 0.0, sum_golden = 0.0;
    for (int b = 0; b < GRID; ++b) sum_partial += (double)hOut[b];
    for (int i = 0; i < n; ++i)    sum_golden  += (double)hIn[i];

    printf("\n【最终全局和 = host 二次归约 Σ output[0..%d)】\n", GRID);
    printf("   Σ 局部和 = %.6f\n", sum_partial);
    printf("   Σ input  = %.6f  (CPU 全量 golden)\n", sum_golden);
    printf("   差       = %.3e\n", fabs(sum_partial - sum_golden));

    // ---------- 校验 ----------
    int bad = 0, nan_hole = 0;
    double max_abs = 0.0;
    for (int b = 0; b < GRID; ++b) {
        if (std::isnan(hOut[b])) { nan_hole++; continue; }
        int lo = b * BLOCK, hi = std::min(n, lo + BLOCK);
        double g = 0.0;
        for (int i = lo; i < hi; ++i) g += (double)hIn[i];
        double d = fabs((double)hOut[b] - g);
        max_abs = std::max(max_abs, d);
        if (d > 1e-5) bad++;
    }

    // 带宽：读 n*4 字节，写 GRID*4 字节
    double gbps = 4.0 * (n + GRID) / (ms * 1e6);

    printf("\n------------------------------------------------------------\n");
    printf("校验① 各 block 局部和 vs golden : bad=%d, max_abs=%.3e %s\n",
           bad, max_abs, bad ? "❌" : "✅");
    printf("校验② 全局和 Σoutput vs Σinput  : diff=%.3e %s\n",
           fabs(sum_partial - sum_golden),
           fabs(sum_partial - sum_golden) < 1e-5 ? "✅" : "❌");
    printf("校验③ output 漏写(nan 残留)    : %d %s\n",
           nan_hole, nan_hole ? "❌" : "✅");
    printf("耗时: %.4f ms (%d 次平均) | 有效带宽: %.2f GB/s\n", ms, reps, gbps);
    printf("============================================================\n");

    bool ok = (bad == 0 && nan_hole == 0 &&
               fabs(sum_partial - sum_golden) < 1e-5);
    printf("总体: %s\n", ok ? "✅ 结果正确" : "❌ 存在错误");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dIn)); CUDA_CHECK(cudaFree(dOut));
    free(hIn); free(hOut);
    return ok ? 0 : 1;
}
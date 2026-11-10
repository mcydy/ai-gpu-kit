// cuda_softmax.cu —— 单文件：你给的 kernel 原样 + main + 输入/输出打印 + 校验
//
// 编译（3090 / SM86）：
//   nvcc -O3 -arch=sm_86 -o cuda_softmax cuda_softmax.cu
// 运行：
//   ./cuda_softmax          # 默认 n=8，全量打印输入 / 输出 / golden
//   ./cuda_softmax 16       # n=16
//   ./cuda_softmax 1024     # 单 block 上限（n 必须 <= 1024）
//
// ★ 这个 kernel 是【单 block】归并：max 和 sum 都是块内局部量，
//   grid 必须为 1、blockDim >= n 且为 2 的幂。长序列要两遍 kernel
//   或 online softmax（跨块用 exp(m_old - m_new) 修正）。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>          // FLT_MAX（你 kernel 里用了 -FLT_MAX）
#include <algorithm>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ==================== 唯一的 kernel（你给的原版，一行未改）====================
__global__ void softmax(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < n) ? input[idx] : -FLT_MAX;

    // Step 1: 块内最大值归约
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) 
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Step 2: exp(x - max) + 求和归约
    float exp_val = expf(val - max_val);
    sdata[tid] = exp_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];

    // Step 3: 归一化写出
    if (idx < n)
        output[idx] = exp_val / sum;
}
// ============================================================================

// ---------- 打印 1D 数组的窗口，带下标 ----------
static void print_vec(const char* name, const float* v,
                      int start, int len, int valid_n) {
    printf("%s  下标[%d..%d]:\n", name, start, start + len - 1);
    for (int s = start; s < start + len; s += 8) {
        int e = std::min(s + 8, start + len);
        printf("   idx:");
        for (int i = s; i < e; ++i) printf("%8d", i);
        printf("\n   val:");
        for (int i = s; i < e; ++i) {
            if (i >= valid_n)          printf("     ---");
            else if (std::isnan(v[i])) printf("     nan");   // 没被 kernel 写入
            else                       printf("%8.5f", v[i]);
        }
        printf("\n");
    }
}

// ---------- CPU 全局 softmax golden（同样的「减 max」稳定算法）----------
static void cpu_softmax(const float* in, float* out, int n) {
    float mx = -FLT_MAX;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, in[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)expf(in[i] - mx);
    for (int i = 0; i < n; ++i) out[i] = (float)(expf(in[i] - mx) / sum);
}

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 8;
    if (n <= 0) { fprintf(stderr, "n 必须为正整数\n"); return 1; }

    // ★ blockDim：>= n 的最小 2 的幂（树形归约要求 2 的幂）
    int BLOCK = 1;
    while (BLOCK < n) BLOCK <<= 1;
    if (BLOCK > 1024) {
        fprintf(stderr,
            "n=%d 超过单 block 上限 1024：这个 kernel 的 max/sum 是块内局部量，\n"
            "  多 block 会得到分块 softmax（全局 Σ != 1）。长序列请改两遍 kernel\n"
            "  或 online softmax。\n", n);
        return 1;
    }
    const int W = std::min(n, 16);           // 打印窗口
    const size_t bytes = (size_t)n * sizeof(float);
    const size_t shmem = (size_t)BLOCK * sizeof(float);   // ★ extern __shared__ 必须靠 launch 第三参数传

    printf("============================================================\n");
    printf("softmax (单 block 归并) | n=%d | blockDim=%d | grid=1 | shared=%.1f KB\n",
           n, BLOCK, 1, shmem / 1024.0);
    printf("有效线程=%d / 启动=%d (%.1f%%)，越界线程填 -FLT_MAX → exp→0，不污染 sum\n",
           n, BLOCK, 100.0 * n / BLOCK);
    printf("打印窗口=%d 个元素\n", W);
    printf("============================================================\n\n");

    // ---------- 输入：-3..3，故意留一个可控范围，看得出 softmax 分布 ----------
    // 想验证「减 max 防溢出」：把下面 hIn[0] 改成 1000.0f，
    //   expf(1000) 会是 inf，但 kernel 先减 max(=1000) 后 expf(0)=1，结果仍然正确。
    float *hIn  = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);
    if (!hIn || !hOut || !hRef) { fprintf(stderr, "host malloc 失败\n"); return 1; }
    for (int i = 0; i < n; ++i) {
        hIn[i]  = (float)((i % 7) - 3);
        hOut[i] = NAN;                       // canary：检测漏写
    }
    cpu_softmax(hIn, hRef, n);

    print_vec("【输入 input】", hIn, 0, W, n);
    printf("\n");

    // ---------- device ----------
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc((void**)&dIn,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn,  hIn,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOut, hOut, bytes, cudaMemcpyHostToDevice));  // NAN 带下去

    dim3 grid(1);          // ★ 必须 1，理由见文件头
    dim3 block(BLOCK);

    // warmup，吃掉首次 CUDA 上下文开销
    softmax<<<grid, block, shmem>>>(dIn, dOut, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    softmax<<<grid, block, shmem>>>(dIn, dOut, n);   // ★ 计时期间 kernel 内不 printf
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    // ---------- 打印输出 + golden ----------
    print_vec("【输出 output】", hOut, 0, W, n);
    printf("\n");
    print_vec("【CPU golden · 全局 softmax】", hRef, 0, W, n);

    // ---------- 校验 ----------
    int bad = 0, nan_hole = 0;
    double max_abs = 0.0, sum_out = 0.0, sum_ref = 0.0;
    float min_v = 1.0f, max_v = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(hOut[i])) { nan_hole++; continue; }
        double diff = fabs((double)hOut[i] - (double)hRef[i]);
        max_abs = std::max(max_abs, diff);
        if (diff > 1e-6) bad++;
        sum_out += hOut[i];
        sum_ref += hRef[i];
        min_v = fminf(min_v, hOut[i]);
        max_v = fmaxf(max_v, hOut[i]);
    }

    printf("\n------------------------------------------------------------\n");
    printf("校验① 全量 vs golden   : bad=%d, max_abs=%.3e %s\n",
           bad, max_abs, bad ? "❌" : "✅");
    printf("校验② Σ output         : %.6f  (应 = 1.0) %s\n",
           sum_out, fabs(sum_out - 1.0) < 1e-5 ? "✅" : "❌");
    printf("     Σ golden          : %.6f\n", sum_ref);
    printf("校验③ 概率范围         : [%.6f, %.6f] %s\n",
           min_v, max_v, (min_v >= 0.0f && max_v <= 1.0f) ? "✅" : "❌");
    printf("校验④ 合法区 nan(漏写) : %d %s\n", nan_hole, nan_hole ? "❌" : "✅");
    printf("耗时: %.4f ms  (n=%d，单 block 小 kernel，数字仅供参考)\n", ms, n);
    printf("============================================================\n");

    bool ok = (bad == 0 && nan_hole == 0 && fabs(sum_out - 1.0) < 1e-5);
    printf("总体: %s\n", ok ? "✅ 结果正确" : "❌ 存在错误");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dIn)); CUDA_CHECK(cudaFree(dOut));
    free(hIn); free(hOut); free(hRef);
    return ok ? 0 : 1;
}
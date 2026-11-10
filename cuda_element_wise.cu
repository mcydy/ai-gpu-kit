// cuda_element_wise.cu —— 单文件：最简 vector add + main + 输入/输出打印 + 校验 + 计时
//
// 编译（3090 / SM86）：
//   nvcc -O3 -arch=sm_86 -o cuda_element_wise cuda_element_wise.cu
// 运行：
//   ./cuda_element_wise          # 默认 n=1024，打印窗口 8 个元素
//   ./cuda_element_wise 8        # ★ 小尺寸：a / b / c 全量打印
//   ./cuda_element_wise 257      # 非 256 倍数：看 grid 多起一块 + idx<n 边界 mask + guard 区
//   ./cuda_element_wise 4097 16  # 指定打印窗口元素个数
//
// c[i] = a[i] + b[i]，1D 数组，fp32。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define BLOCK 256   // 1D block：256 线程（warp=32 的整数倍，CUDA 最常见取值）

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
__global__ void vec_add(float* a, float* b, float* c, int n) {
    // 1D 索引：blockIdx.x 是第几块，blockDim.x 是每块线程数，
    // threadIdx.x 是块内编号 → 三者拼出全局唯一的元素下标。
    // 元素是 1D 数组，所以用 1D grid/block 就够了，不需要 blockIdx.y。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ★ 边界 mask：n 不是 BLOCK 整数倍时，最后一块里 idx>=n 的线程不能读写，
    //   否则会越界读 a/b、越界写 c，踩坏相邻内存。
    if (idx < n) c[idx] = a[idx] + b[idx];
}
// ============================================================================

// ---------- 打印 1D 数组的一段窗口，带下标；nan 用 canary 标记 ----------
static void print_vec(const char* name, const float* v,
                      int start, int len, int valid_n) {
    printf("%s  下标[%d..%d]:\n", name, start, start + len - 1);
    for (int s = start; s < start + len; s += 8) {
        int e = std::min(s + 8, start + len);
        printf("   idx:");
        for (int i = s; i < e; ++i) printf("%8d", i);
        printf("\n   val:");
        for (int i = s; i < e; ++i) {
            if (i >= valid_n)         printf("    ---");        // guard 区（合法数组之外）
            else if (std::isnan(v[i])) printf("     nan");       // 没被 kernel 写入
            else                       printf("%8.4f", v[i]);
        }
        printf("\n");
    }
}

// ---------- 同一窗口的 golden：a[i]+b[i]（fp32，数据可精确表示）----------
static void print_golden(const char* name, const float* a, const float* b,
                         int start, int len, int valid_n) {
    printf("%s (CPU 真值 a+b)  下标[%d..%d]:\n", name, start, start + len - 1);
    for (int s = start; s < start + len; s += 8) {
        int e = std::min(s + 8, start + len);
        printf("   idx:");
        for (int i = s; i < e; ++i) printf("%8d", i);
        printf("\n   val:");
        for (int i = s; i < e; ++i) {
            if (i >= valid_n) printf("    ---");
            else              printf("%8.4f", a[i] + b[i]);
        }
        printf("\n");
    }
}

// ---------- 计时：warmup + reps 次取平均 ----------
template <typename Launch>
static float bench_ms(Launch launch, int reps) {
    launch();                                   // warmup：吃掉首次 CUDA 上下文/启动开销
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
    const int n = (argc > 1) ? atoi(argv[1]) : 1024;
    if (n <= 0) { fprintf(stderr, "n 必须为正整数\n"); return 1; }
    const int W_req = (argc > 2) ? atoi(argv[2]) : (n <= 16 ? n : 8);
    const int W = std::max(1, std::min(W_req, n));

    // cdiv 向上取整：n=257 时 GRID=2，会多起一整块（512 个线程）
    const int GRID = (n + BLOCK - 1) / BLOCK;
    const int padded = GRID * BLOCK;            // 启动线程总数
    const int guard = padded - n;               // ★ 多出来的"虚拟线程"数
    const size_t bytes = (size_t)padded * sizeof(float);
    const int reps = (n <= 4096) ? 20 : 5;

    printf("============================================================\n");
    printf("vec_add: c[i] = a[i] + b[i]  (1D, fp32)\n");
    printf("n=%d | BLOCK=%d | GRID=%d | 启动线程=%d, 有效=%d (%.1f%%), guard=%d\n",
           n, BLOCK, GRID, padded, n, 100.0 * n / padded, guard);
    printf("打印窗口=%d 个元素 | reps=%d\n", W, reps);
    printf("============================================================\n\n");

    // ---------- host：按 padded 分配，多出来的 guard 区专门检测越界写 ----------
    // 数据填 1/16 的整数倍 → fp32 可精确表示，a+b 无舍入，校验能做到逐 bit 相等
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc 失败\n"); return 1; }
    for (int i = 0; i < padded; ++i) {
        hA[i] = (float)((i % 19) - 9) * 0.0625f;
        hB[i] = (float)((i % 23) - 11) * 0.0625f;
        hC[i] = NAN;                            // ★ canary：合法位没被写 / guard 被误写都能看出来
    }

    // ---------- 打印输入 ----------
    print_vec("【输入 a】", hA, 0, W, n);
    printf("\n");
    print_vec("【输入 b】", hB, 0, W, n);
    printf("\n");

    // ---------- device：HBM 分配 + PCIe 搬运 ----------
    // 数据通路：HBM(a,b) → L2/L1 → 寄存器(做加法) → 直接 global store 回 HBM(c)
    //           GPU 是 load-store 架构，没有"内存到内存"指令，必须经寄存器过一遍
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dB, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, bytes, cudaMemcpyHostToDevice));  // NAN 也带下去

    dim3 block(BLOCK);      // 1D：block.x = 256
    dim3 grid(GRID);        // 1D：grid.x  = cdiv(n, 256)

    // ★ 计时期间 kernel 内不做任何 printf（device printf 会拖垮带宽数字）
    float ms = bench_ms([&]() {
        vec_add<<<grid, block>>>(dA, dB, dC, n);
    }, reps);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // ---------- 打印输出：头部窗口 + 跨越 n 的尾部窗口（看 mask 与 guard）----------
    print_vec("【输出 c —— 头部】", hC, 0, W, n);
    printf("\n");
    print_golden("【CPU 真值 —— 头部】", hA, hB, 0, W, n);

    if (n > W || guard > 0) {
        int tail_start = std::max(0, n - W);
        int tail_len = std::min(padded - tail_start, W + std::min(guard, 8));
        printf("\n【输出 c —— 尾部 + guard 区】(idx>=%d 应显示 ---，即未被写入)\n", n);
        print_vec("C tail", hC, tail_start, tail_len, n);
        printf("\n");
        print_golden("golden tail", hA, hB, tail_start, tail_len, n);
    }

    // ---------- 校验 ----------
    // ① 合法区全量对 a+b；② 合法区不能有 nan（漏写）；③ guard 区必须仍是 nan（越界写）
    int bad = 0, nan_hole = 0, guard_written = 0;
    double max_abs = 0.0;
    for (int i = 0; i < padded; ++i) {
        if (i < n) {
            if (std::isnan(hC[i])) { nan_hole++; continue; }
            double diff = fabs((double)hC[i] - ((double)hA[i] + (double)hB[i]));
            max_abs = std::max(max_abs, diff);
            if (diff > 1e-6) bad++;
        } else {
            if (!std::isnan(hC[i])) guard_written++;   // mask 漏了就会在这里被抓到
        }
    }

    // 带宽：读 a、b 各 n*4 字节，写 c 为 n*4 字节 → 共 12n 字节
    // vec_add 是纯 memory-bound 算子，用 GB/s 衡量比 GFLOPS 有意义
    double gbps = 12.0 * n / (ms * 1e6);

    printf("\n------------------------------------------------------------\n");
    printf("校验① 全量 vs a+b         : bad=%d, max_abs=%.3e %s\n",
           bad, max_abs, bad ? "❌" : "✅");
    printf("校验② 合法区 nan 残留(漏写): %d %s\n",
           nan_hole, nan_hole ? "❌" : "✅");
    printf("校验③ guard 区被写(越界)  : %d %s\n",
           guard_written, guard_written ? "❌" : "✅");
    printf("耗时: %.4f ms (%d 次平均) | 有效带宽: %.2f GB/s\n", ms, reps, gbps);
    printf("提示: 每线程只做 1 次加法却要读 8B 写 4B，算术强度极低，\n"
           "      纯带宽瓶颈；n 很大时带宽会逼近显存上限。\n");
    printf("============================================================\n");

    bool ok = (bad == 0 && nan_hole == 0 && guard_written == 0);
    printf("总体: %s\n", ok ? "✅ 结果正确" : "❌ 存在错误");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return ok ? 0 : 1;
}
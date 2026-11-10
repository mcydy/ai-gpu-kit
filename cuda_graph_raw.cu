/*
 * 最底层版本：手写 __global__ kernel + 手工构建 CUDA Graph
 *
 * 与 Python/Triton 版的区别：
 *   Triton 版    —— 你能看到 kernel 源码和 PTX，但图是 torch.cuda.graph() 帮你捕获的
 *   本    版     —— 你亲手调用 cudaGraphCreate / cudaGraphAddKernelNode 把节点
 *                  一个个塞进图里，还能用 cudaGraphGetNodes / GetEdges 把图的
 *                  节点数、依赖边数打印出来，看见"图"作为数据结构的真身
 *
 * 编译（arch 按你的卡改，见文件末尾说明）：
 *   nvcc -O2 -arch=sm_86 -o cuda_graph_raw cuda_graph_raw.cu
 *   ./cuda_graph_raw
 */

/*
 * 编译与运行
 * ──────────────────────────────────────────────────────────────
 *   # 先查你的 GPU 计算能力（/arch 参数要对，否则可能报 no kernel image）
 *   nvidia-smi --query-gpu=compute_cap --format=csv
 *   # 或者让 nvcc 自动检测（部分版本支持）
 *   nvcc -O2 -arch=native -o cuda_graph_raw cuda_graph_raw.cu
 *
 *   # 常见 arch 对照：
 *   #   sm_75  RTX 20xx / T4
 *   #   sm_80  A100
 *   #   sm_86  RTX 30xx (含你的 3090 一类)
 *   #   sm_89  RTX 40xx
 *   #   sm_90  H100
 *
 *   ./cuda_graph_raw
 *
 * 预期看到的现象
 * ──────────────────────────────────────────────────────────────
 *   1. 图结构打印：节点数 = CHAIN，依赖边数 = CHAIN-1（一条串行链）
 *      —— 这证明"图"是真实存在的数据结构，不是抽象概念
 *   2. 性能：graph 明显快于 eager，且 kernel 越多差距越大
 *   3. 三种执行路径结果一致，证明 Graph 不改变计算语义，只改变提交方式
 *
 * 想要更极端的效果，把 N 调到 256、CHAIN 调到 100 再跑，
 * 此时 GPU 几乎瞬间算完，时间全花在 CPU 提交上，Graph 收益会非常夸张。
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(e));                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ════════════════════════════════════════════════════════════════
// 1. kernel 本体 —— 这就是 GPU 上跑的函数，清清楚楚看得见
// ════════════════════════════════════════════════════════════════
__global__ void tiny_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(in[i] + 1.0f, 0.0f);
    }
}

__global__ void scale_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 0.5f;
    }
}

// 汇聚节点：把两个并行分支的结果加起来
__global__ void join_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// ════════════════════════════════════════════════════════════════
// 2. 计时工具
// ════════════════════════════════════════════════════════════════
float bench(int iters, int warmup, void (*fn)(cudaStream_t), cudaStream_t s) {
    for (int i = 0; i < warmup; i++) fn(s);
    CHECK(cudaStreamSynchronize(s));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, s));
    for (int i = 0; i < iters; i++) fn(s);
    CHECK(cudaEventRecord(stop, s));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / iters * 1000.0f;      // 微秒
}

// ════════════════════════════════════════════════════════════════
// 3. 全局配置
// ════════════════════════════════════════════════════════════════
static const int N        = 4096;     // 数据量故意取小：让 kernel 极快，凸显 launch 开销
static const int BLOCK    = 256;
static const int GRID     = (N + BLOCK - 1) / BLOCK;
static const int CHAIN    = 20;       // 一条链上的 kernel 数量

static float *d_in, *d_a, *d_b;       // 显存 buffer
static cudaGraphExec_t g_exec_manual; // 手工构建的图
static cudaGraphExec_t g_exec_capture;// capture 得到的图
static cudaGraph_t     g_graph_manual;

// ── eager：CPU 每次循环提交一个 kernel ──────────────────────────
static void run_eager(cudaStream_t s) {
    for (int i = 0; i < CHAIN; i++) {
        const float* src = (i % 2 == 0) ? d_in : d_a;
        float*       dst = (i % 2 == 0) ? d_a  : d_b;
        tiny_kernel<<<GRID, BLOCK, 0, s>>>(dst, src, N);
    }
}

// ── graph replay：CPU 只提交一次，GPU 一口气跑完整张图 ──────────
static void run_graph(cudaStream_t s) {
    CHECK(cudaGraphLaunch(g_exec_capture, s));
}

static void run_graph_manual(cudaStream_t s) {
    CHECK(cudaGraphLaunch(g_exec_manual, s));
}

// ════════════════════════════════════════════════════════════════
// 4. 打印图的结构 —— 让你看见"图"真的是一张图
// ════════════════════════════════════════════════════════════════
static void describe_graph(cudaGraph_t graph, const char* name) {
    size_t num_nodes = 0, num_edges = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));

    std::vector<cudaGraphNode_t> nodes(num_nodes);
    CHECK(cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

    CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges));
    std::vector<cudaGraphNode_t> from(num_edges), to(num_edges);
    if (num_edges > 0) {
        CHECK(cudaGraphGetEdges(graph, from.data(), to.data(), &num_edges));
    }

    printf("  [%s] 节点数 = %zu，依赖边数 = %zu\n", name, num_nodes, num_edges);

    // 统计各节点类型，看看图里都装了些什么
    size_t kernel_cnt = 0;
    for (size_t i = 0; i < num_nodes; i++) {
        cudaGraphNodeType t;
        CHECK(cudaGraphNodeGetType(nodes[i], &t));
        if (t == cudaGraphNodeTypeKernel) kernel_cnt++;
    }
    printf("          其中 kernel 节点 = %zu，链式依赖边 = %zu（串行链应为 N-1）\n",
           kernel_cnt, num_edges);
}

// ════════════════════════════════════════════════════════════════
// 5. 手工建图：一个节点一个节点塞进去
// ════════════════════════════════════════════════════════════════
static void build_graph_manual() {
    CHECK(cudaGraphCreate(&g_graph_manual, 0));

    std::vector<cudaGraphNode_t> nodes(CHAIN);

    for (int i = 0; i < CHAIN; i++) {
        const float* src = (i % 2 == 0) ? d_in : d_a;
        float*       dst = (i % 2 == 0) ? d_a  : d_b;

        // kernel 参数数组：每个元素是指向实参的指针。
        // CUDA 在 AddKernelNode 时会把参数值拷进图里，之后这组局部变量可以安全失效。
        void* args[] = {(void*)&dst, (void*)&src, (void*)&N};

        cudaKernelNodeParams p = {};
        p.func           = (void*)tiny_kernel;
        p.gridDim        = dim3(GRID, 1, 1);
        p.blockDim       = dim3(BLOCK, 1, 1);
        p.sharedMemBytes = 0;
        p.kernelParams   = args;

        // 依赖：第 i 个节点依赖第 i-1 个，形成一条串行链。
        // 想让它并行，这里传空依赖即可 —— 这正是 Graph 能表达 DAG 的地方。
        cudaGraphNode_t* deps      = (i > 0) ? &nodes[i - 1] : nullptr;
        size_t           num_deps  = (i > 0) ? 1 : 0;

        CHECK(cudaGraphAddKernelNode(&nodes[i], g_graph_manual,
                                     deps, num_deps, &p));
    }

    CHECK(cudaGraphInstantiate(&g_exec_manual, g_graph_manual, 0));
}

// ════════════════════════════════════════════════════════════════
// 6. 用流捕获建图：更省事，但你看不见节点是怎么加进去的
// ════════════════════════════════════════════════════════════════
static void build_graph_capture(cudaStream_t s) {
    // 先在普通流上 warmup，确保 kernel 已加载、无隐式初始化动作
    for (int i = 0; i < 3; i++) run_eager(s);
    CHECK(cudaStreamSynchronize(s));

    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    run_eager(s);                       // 这段提交动作被"录"下来，而非真正执行
    CHECK(cudaStreamEndCapture(s, &graph));

    CHECK(cudaGraphInstantiate(&g_exec_capture, graph, 0));
    describe_graph(graph, "stream capture");
    CHECK(cudaGraphDestroy(graph));     // 图已实例化，原始 graph 可以释放
}

// ════════════════════════════════════════════════════════════════
// 7. 额外演示：fork-join DAG（手工图才能直观表达并行分支）
// ════════════════════════════════════════════════════════════════
static void demo_fork_join() {
    printf("\n── 附加演示：fork-join DAG（手工图才能显式表达并行）──\n");

    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));

    cudaGraphNode_t n0, n1, n2;

    // 分支 A：d_in -> d_a
    {
        float* dst = d_a; const float* src = d_in; int n = N;
        void* args[] = {(void*)&dst, (void*)&src, (void*)&n};
        cudaKernelNodeParams p = {};
        p.func = (void*)tiny_kernel;
        p.gridDim = dim3(GRID, 1, 1);
        p.blockDim = dim3(BLOCK, 1, 1);
        p.kernelParams = args;
        CHECK(cudaGraphAddKernelNode(&n0, graph, nullptr, 0, &p));
    }

    // 分支 B：d_in -> d_b（不依赖 n0，因此可与 n0 并行执行）
    {
        float* dst = d_b; const float* src = d_in; int n = N;
        void* args[] = {(void*)&dst, (void*)&src, (void*)&n};
        cudaKernelNodeParams p = {};
        p.func = (void*)scale_kernel;
        p.gridDim = dim3(GRID, 1, 1);
        p.blockDim = dim3(BLOCK, 1, 1);
        p.kernelParams = args;
        CHECK(cudaGraphAddKernelNode(&n1, graph, nullptr, 0, &p));
    }

    // 汇聚：d_a = d_a + d_b，同时依赖 n0 与 n1
    {
        float* out = d_a; float* a = d_a; float* b = d_b; int n = N;
        void* args[] = {(void*)&out, (void*)&a, (void*)&b, (void*)&n};
        cudaKernelNodeParams p = {};
        p.func = (void*)join_kernel;
        p.gridDim = dim3(GRID, 1, 1);
        p.blockDim = dim3(BLOCK, 1, 1);
        p.kernelParams = args;
        cudaGraphNode_t deps[] = {n0, n1};
        CHECK(cudaGraphAddKernelNode(&n2, graph, deps, 2, &p));
    }

    describe_graph(graph, "fork-join");
    printf("  结构：n0 ─┐\n");
    printf("           ├→ n2        （n0 与 n1 之间无边 → 可并行）\n");
    printf("       n1 ─┘\n");

    // 真跑一次，验证 DAG 语义正确
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    std::vector<float> h_in(N, 2.0f);
    CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaStreamSynchronize(0));

    std::vector<float> got(N);
    CHECK(cudaMemcpy(got.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost));
    float expect = (2.0f + 1.0f) + (2.0f * 0.5f);   // tiny: x+1 ; scale: x*0.5
    printf("  执行结果 d_a[0] = %.3f，期望 %.3f  %s\n",
           got[0], expect, fabsf(got[0] - expect) < 1e-4 ? "OK" : "MISMATCH!");

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("配置: N=%d  BLOCK=%d  GRID=%d  链长 CHAIN=%d\n\n", N, BLOCK, GRID, CHAIN);

    // ── 显存准备 ──
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_a,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_b,  N * sizeof(float)));

    std::vector<float> h(N, 1.0f);
    CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    // ── 建图（两种）──
    build_graph_manual();
    describe_graph(g_graph_manual, "手工构建");
    build_graph_capture(s);

    // ── 正确性校验：三种跑法结果必须一致 ──
    run_eager(s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> ref(N);
    CHECK(cudaMemcpy(ref.data(), (CHAIN % 2 == 0) ? d_b : d_a,
                     N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    run_graph(s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> got1(N);
    CHECK(cudaMemcpy(got1.data(), (CHAIN % 2 == 0) ? d_b : d_a,
                     N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    run_graph_manual(s);
    CHECK(cudaStreamSynchronize(s));
    std::vector<float> got2(N);
    CHECK(cudaMemcpy(got2.data(), (CHAIN % 2 == 0) ? d_b : d_a,
                     N * sizeof(float), cudaMemcpyDeviceToHost));

    float maxerr = 0.0f;
    for (int i = 0; i < N; i++) {
        maxerr = fmaxf(maxerr, fabsf(ref[i] - got1[i]));
        maxerr = fmaxf(maxerr, fabsf(ref[i] - got2[i]));
    }
    printf("\n正确性校验：eager / capture / 手工图 三者最大误差 = %.6f  %s\n",
           maxerr, maxerr < 1e-5 ? "OK" : "MISMATCH!");

    // ── 性能对比 ──
    const int ITERS = 500, WARM = 50;
    float t_eager  = bench(ITERS, WARM, run_eager, s);
    float t_cap    = bench(ITERS, WARM, run_graph, s);
    float t_manual = bench(ITERS, WARM, run_graph_manual, s);

    printf("\n" "── 性能对比（%d 个 kernel 的串行链）" "──\n", CHAIN);
    printf("  eager 逐个提交     : %8.2f μs   (单 kernel 平均 %.2f μs)\n",
           t_eager, t_eager / CHAIN);
    printf("  graph(capture)     : %8.2f μs   (单 kernel 平均 %.2f μs)  → %.2fx\n",
           t_cap, t_cap / CHAIN, t_eager / t_cap);
    printf("  graph(手工构建)    : %8.2f μs   (单 kernel 平均 %.2f μs)  → %.2fx\n",
           t_manual, t_manual / CHAIN, t_eager / t_manual);

    printf("\n读法：\n");
    printf("  · eager 里单 kernel 的耗时 ≈ CPU 提交开销 + GPU 执行时间\n");
    printf("  · graph 里单 kernel 的耗时 ≈ 仅 GPU 执行时间（提交开销被摊平）\n");
    printf("  · 两者之差，就是被 CUDA Graph 干掉的驱动开销\n");
    printf("  · capture 与手工构建性能应基本一致，因为最终都是同一份 cudaGraphExec\n");

    // ── fork-join DAG 演示 ──
    demo_fork_join();

    // ── 清理 ──
    CHECK(cudaGraphExecDestroy(g_exec_manual));
    CHECK(cudaGraphExecDestroy(g_exec_capture));
    CHECK(cudaGraphDestroy(g_graph_manual));
    CHECK(cudaStreamDestroy(s));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    return 0;
}


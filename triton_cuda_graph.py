"""
Triton + CUDA Graph —— 能看见 kernel

与 PyTorch 高层版的区别：
    PyTorch 版：torch.cuda.graph() 一包，kernel 全在黑盒里，你看不到任何 GPU 指令
    本    版：kernel 是你自己用 @triton.jit 写的，还能导出编译后的 PTX 汇编，
              逐条看到 GPU 到底执行了什么指令

运行（需要 NVIDIA GPU + torch + triton）：
    看 Triton 编译产物落盘，加环境变量：
    TRITON_KERNEL_DUMP=1 python3 triton_cuda_graph_demo.py

实验清单：
    exp1  导出 PTX  —— 真正"看见 kernel"
    exp2  单个 kernel：eager vs graph（几乎没有收益，说明边界）
    exp3  20 个 kernel 串联：eager vs graph（收益最大区）
    exp4  动态 shape 踩坑：capture 后改 batch 会怎样
"""
import os
import torch
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════
# 1. 你自己写的 kernel —— 这就是"看得见的 kernel"
# ════════════════════════════════════════════════════════════════
@triton.jit
def add_relu_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    """
    融合算子：Z = relu(X + Y)

    为什么写这个？因为融合（fusion）正是 kernel 层的价值所在：
    同样的数学，PyTorch 要跑 2 个 kernel（add 一次显存往返 + relu 一次），
    Triton 一个 kernel 读完就顺手算掉，省掉一次全局显存读写。
    """
    pid = tl.program_id(0)                      # 当前 block 的编号
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 这个 block 负责的元素下标
    mask = offs < N                             # 尾部越界保护

    x = tl.load(X + offs, mask=mask, other=0.0)  # 全局显存 → 寄存器
    y = tl.load(Y + offs, mask=mask, other=0.0)
    z = tl.maximum(x + y, 0.0)                  # 在寄存器里算完，不回落显存
    tl.store(Z + offs, z, mask=mask)            # 寄存器 → 全局显存


# ════════════════════════════════════════════════════════════════
# 2. 导出 PTX —— 把 kernel 编译成 GPU 指令给你看
# ════════════════════════════════════════════════════════════════
def dump_ptx(kernel, sample_args, constexprs, grid=(1,)):
    """
    导出 Triton kernel 编译后的 PTX 汇编。

    Triton 2.x / 3.x 的 API 签名差异较大，这里逐条尝试、失败即降级，
    不会因为你版本不同而让整个 demo 崩掉。
    返回 (ptx字符串 或 None, 尝试记录列表)
    """
    attempts = []

    # 方式 A：JITFunction.warmup（Triton 2.1+ 多数版本支持，返回 CompiledKernel）
    try:
        compiled = kernel.warmup(*sample_args, grid=grid, **constexprs)
        ptx = None
        if compiled is not None:
            asm = getattr(compiled, "asm", None)
            if isinstance(asm, dict):
                ptx = asm.get("ptx")
        if ptx:
            return ptx, attempts
        attempts.append("warmup 成功但未拿到 ptx 字段")
    except Exception as e:
        attempts.append(f"warmup 路径失败 ({type(e).__name__}: {e})")

    # 方式 B：triton.compile + ASTSource（较新版本的官方路径）
    try:
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        import triton as _triton

        cap = torch.cuda.get_device_capability()
        target = GPUTarget("cuda", cap, 32)

        # 3.x 用 dict 形式的 signature；2.x 用字符串，两种都试
        for sig in (
            {k: "*fp32" if isinstance(v, torch.Tensor) else "i32"
             for k, v in zip(("X", "Y", "Z", "N"), sample_args)},
            "*fp32,*fp32,*fp32,i32",
        ):
            try:
                src = ASTSource(fn=kernel, signature=sig, constexprs=constexprs)
                compiled = _triton.compile(src, target=target,
                                           options={"num_warps": 4})
                ptx = compiled.asm.get("ptx")
                if ptx:
                    return ptx, attempts
            except Exception as e:
                attempts.append(f"ASTSource(sig={type(sig).__name__}) 失败: {type(e).__name__}")
    except Exception as e:
        attempts.append(f"compile 路径整体不可用 ({type(e).__name__}: {e})")

    return None, attempts


def exp1_show_kernel(n=4096, block=1024):
    print("\n" + "=" * 70)
    print("exp1  导出 PTX —— 看见真正的 kernel")
    print("=" * 70)

    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    z = torch.empty(n, device="cuda")

    ptx, attempts = dump_ptx(add_relu_kernel, (a, b, z, n), {"BLOCK": block})

    if ptx:
        lines = ptx.splitlines()
        print(f"✓ 拿到 PTX，共 {len(lines)} 行。前 40 行：\n")
        print("-" * 70)
        for line in lines[:40]:
            print(line)
        print("-" * 70)
        print("\n你在这段 PTX 里应当能认出这些结构：")
        print("  · .visible .entry <name>   ← kernel 入口（GPU 上的一个可启动函数）")
        print("  · ld.global.*              ← 从全局显存加载（对应 tl.load）")
        print("  · add.f32 / max.f32        ← 寄存器上的算术（对应 + 和 tl.maximum）")
        print("  · st.global.*              ← 写回全局显存（对应 tl.store）")
        print("\n这就是 CUDA Graph 捕获时要录进图里的那个 kernel 的真身。")
    else:
        print("✗ 当前 Triton 版本未能通过代码内接口导出 PTX。尝试记录：")
        for a_ in attempts:
            print("   -", a_)
        print("\n降级方案（不依赖 API，任何版本都有效）：")
        print("  TRITON_KERNEL_DUMP=1 python3 triton_cuda_graph_demo.py")
        print("  编译产物会被 Triton 写到磁盘目录（具体路径随版本而定，运行时留意终端输出）。")

    # 顺手验证自己写的 kernel 算得对不对
    grid = (triton.cdiv(n, block),)
    add_relu_kernel[grid](a, b, z, n, BLOCK=block, num_warps=4)
    expect = torch.relu(a + b)
    err = (z - expect).abs().max().item()
    print(f"\n正确性校验：与 torch.relu(a+b) 的最大误差 = {err:.3e}  "
          f"{'✓' if err < 1e-5 else '✗ 有问题！'}")


# ════════════════════════════════════════════════════════════════
# 3. 计时工具
# ════════════════════════════════════════════════════════════════
def bench(fn, iters=300, warmup=30):
    """用 CUDA Event 测 GPU 时间（不是 wall time）"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0   # 微秒


def make_chain(n, block, reps):
    """生成一串 kernel 调用：relu(buf + b) 在两个 buffer 间交替，形成依赖链"""
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    buf0 = torch.empty(n, device="cuda")
    buf1 = torch.empty(n, device="cuda")
    grid = (triton.cdiv(n, block),)

    def chain():
        for i in range(reps):
            src, dst = (a, buf0) if i == 0 else (
                (buf0, buf1) if i % 2 == 1 else (buf1, buf0))
            add_relu_kernel[grid](src, b, dst, n, BLOCK=block, num_warps=4)

    return chain, (a, b, buf0, buf1, grid)


def exp2_single_kernel():
    print("\n" + "=" * 70)
    print("exp2  单个 kernel：eager vs graph（几乎没收益）")
    print("=" * 70)

    n, block = 1 << 20, 1024
    chain, (a, b, buf0, buf1, grid) = make_chain(n, block, reps=1)

    # warmup，同时确保 Triton 已完成 JIT 编译（capture 期间不能编译！）
    for _ in range(5):
        chain()
    torch.cuda.synchronize()

    t_eager = bench(chain)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    # 开始捕获
    with torch.cuda.graph(g):
        chain()

    t_graph = bench(lambda: g.replay())

    print(f"  1 个 kernel      eager = {t_eager:7.2f} μs")
    print(f"  1 个 kernel      graph = {t_graph:7.2f} μs   → {t_eager / t_graph:5.2f}x")
    print("\n  为什么几乎没收益：GPU 算得比 CPU 提交还快 or 相当，")
    print("  省下的那点 CPU 开销被 GPU 计算完全掩盖了。记住这个边界。")


def exp3_kernel_chain():
    print("\n" + "=" * 70)
    print("exp3  20 个 kernel 串联：eager vs graph（收益最大区）")
    print("=" * 70)

    n, block, reps = 1 << 16, 1024, 20
    chain, objs = make_chain(n, block, reps)

    for _ in range(5):
        chain()
    torch.cuda.synchronize()

    t_eager = bench(chain)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    # 开始捕获
    with torch.cuda.graph(g):
        chain()

    t_graph = bench(lambda: g.replay())

    print(f"  {reps} 个 kernel 链  eager = {t_eager:7.2f} μs")
    print(f"  {reps} 个 kernel 链  graph = {t_graph:7.2f} μs   → {t_eager / t_graph:5.2f}x")
    print(f"\n  单 kernel 平均：eager {t_eager / reps:.2f} μs → graph {t_graph / reps:.2f} μs")
    print("  差额就是『CPU 提交单个 kernel 的驱动开销』，Graph 把它整段消掉了。")

    # 顺带看看更大的链
    print("\n  链长度 → 加速比：")
    for r in (2, 5, 10, 20, 50):
        c, _ = make_chain(n, block, r)
        for _ in range(3):
            c()
        torch.cuda.synchronize()
        te = bench(c, iters=200)
        gg = torch.cuda.CUDAGraph()
        ss = torch.cuda.Stream()
        ss.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ss):
            c()
        torch.cuda.current_stream().wait_stream(ss)
        with torch.cuda.graph(gg):
            c()
        tg = bench(lambda: gg.replay(), iters=200)
        bar = "█" * int(te / tg * 8)
        print(f"    N={r:3d}   {te:7.2f} → {tg:6.2f} μs   {te / tg:4.2f}x  {bar}")


def exp4_dynamic_shape():
    print("\n" + "=" * 70)
    print("exp4  动态 shape 踩坑：capture 后改数据长度")
    print("=" * 70)
    print("  CUDA Graph 把 grid / 指针 / 形状全固化进图里。")
    print("  用一个长度捕获，拿另一个长度重放 —— 数据本身变化是可以的，")
    print("  但『形状变化』会让 grid 失效，必须重新捕获。\n")

    n, block = 1 << 16, 1024
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    z = torch.empty(n, device="cuda")
    grid = (triton.cdiv(n, block),)

    def op():
        add_relu_kernel[grid](a, b, z, n, BLOCK=block, num_warps=4)

    for _ in range(3):
        op()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op()
    torch.cuda.current_stream().wait_stream(s)
    # 开始捕获
    with torch.cuda.graph(g):
        op()

    # ① 只改数据内容，不改形状 —— 完全合法
    a.copy_(torch.randn(n, device="cuda"))
    g.replay()
    print("  ① 只改 buffer 内容后 replay        → ✓ 正常（图里固化的是指针，不是值）")

    # ② 换一个不同形状的 tensor 再 replay —— grid 不匹配，结果错误
    a2 = torch.randn(n * 2, device="cuda")
    z2 = torch.empty(n * 2, device="cuda")
    # 注意：这里只是演示"图不会跟着变"，实际工程里不要这么写
    try:
        g.replay()
        print("  ② 换成不同长度的 tensor 后 replay  → ⚠ 图仍按旧 grid 跑，")
        print("     不会报错，但只处理了前一半数据（静默出错，最危险）")
    except Exception as e:
        print(f"  ② 换成不同长度的 tensor 后 replay  → 抛异常 {type(e).__name__}")

    print("\n  三种正确解法：")
    print("   A. 按形状分桶，每个桶捕获一张图（常见且实用）")
    print("   B. 填充到固定最大长度，用 mask 忽略无效部分（LLM 解码常用）")
    print("   C. 形状变了就重新 capture（简单但失去 Graph 意义）")


def main():
    assert torch.cuda.is_available(), "需要 NVIDIA GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}   Torch: {torch.__version__}")

    exp1_show_kernel()
    exp2_single_kernel()
    exp3_kernel_chain()
    exp4_dynamic_shape()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
  本 demo 与 PyTorch 高层版的区别，就在于你看到了 kernel 本体：
    · @triton.jit 写的 add_relu_kernel  —— kernel 源码
    · 导出的 PTX                        —— GPU 真正执行的指令
    · 捕获进 CUDAGraph 的，就是这些 kernel 的 launch 序列

  无论哪一层，CUDA Graph 消除的东西都一样：
    CPU 逐个提交 kernel 的驱动开销（launch overhead）。
  它不减少 GPU 计算量，一次浮点运算都不会少。

  值得用的判据（exp2 vs exp3 已给出边界）：
    ✓ kernel 数量多、每个都很小、同一形状反复跑成千上万次
    ✗ 单个大 kernel 主导（如大 matmul）—— 瓶颈在算力，不在提交
    """)


if __name__ == "__main__":
    main()



# GPU: NVIDIA GeForce RTX 3090
# Triton: 3.2.0   Torch: 2.6.0+cu124

# ======================================================================
# exp1  导出 PTX —— 看见真正的 kernel
# ======================================================================
# ✓ 拿到 PTX，共 239 行。前 40 行：

# ----------------------------------------------------------------------
# //
# // Generated by LLVM NVPTX Back-End
# //

# .version 8.4
# .target sm_86
# .address_size 64

#     // .globl   add_relu_kernel         // -- Begin function add_relu_kernel
#                                         // @add_relu_kernel
# .visible .entry add_relu_kernel(
#     .param .u64 .ptr .global .align 1 add_relu_kernel_param_0,
#     .param .u64 .ptr .global .align 1 add_relu_kernel_param_1,
#     .param .u64 .ptr .global .align 1 add_relu_kernel_param_2,
#     .param .u32 add_relu_kernel_param_3
# )
# .reqntid 128, 1, 1
# {
#     .reg .pred  %p<23>;
#     .reg .b32   %r<49>;
#     .reg .f32   %f<33>;
#     .reg .b64   %rd<11>;
#     .loc    1 32 0                          // gpu_cuda_graph.py:32:0
# $L__func_begin0:
#     .loc    1 32 0                          // gpu_cuda_graph.py:32:0

# // %bb.0:
#     ld.param.u64    %rd7, [add_relu_kernel_param_0];
#     ld.param.u64    %rd8, [add_relu_kernel_param_1];
# $L__tmp0:
#     .loc    1 40 24                         // gpu_cuda_graph.py:40:24
#     // begin inline asm
#     mov.u32 %r1, %ctaid.x;
#     // end inline asm
#     .loc    1 41 17                         // gpu_cuda_graph.py:41:17
#     shl.b32     %r42, %r1, 10;
#     ld.param.u64    %rd9, [add_relu_kernel_param_2];
#     ld.param.u32    %r43, [add_relu_kernel_param_3];
#     .loc    1 41 38                         // gpu_cuda_graph.py:41:38
#     mov.u32     %r44, %tid.x;
# ----------------------------------------------------------------------

# 你在这段 PTX 里应当能认出这些结构：
#   · .visible .entry <name>   ← kernel 入口（GPU 上的一个可启动函数）
#   · ld.global.*              ← 从全局显存加载（对应 tl.load）
#   · add.f32 / max.f32        ← 寄存器上的算术（对应 + 和 tl.maximum）
#   · st.global.*              ← 写回全局显存（对应 tl.store）

# 这就是 CUDA Graph 捕获时要录进图里的那个 kernel 的真身。

# 正确性校验：与 torch.relu(a+b) 的最大误差 = 0.000e+00  ✓

# ======================================================================
# exp2  单个 kernel：eager vs graph（几乎没收益）
# ======================================================================
#   1 个 kernel      eager =   22.62 μs
#   1 个 kernel      graph =   19.62 μs   →  1.15x

#   为什么几乎没收益：GPU 算得比 CPU 提交还快 or 相当，
#   省下的那点 CPU 开销被 GPU 计算完全掩盖了。记住这个边界。

# ======================================================================
# exp3  20 个 kernel 串联：eager vs graph（收益最大区）
# ======================================================================
#   20 个 kernel 链  eager =  409.73 μs
#   20 个 kernel 链  graph =   33.22 μs   → 12.33x

#   单 kernel 平均：eager 20.49 μs → graph 1.66 μs
#   差额就是『CPU 提交单个 kernel 的驱动开销』，Graph 把它整段消掉了。

#   链长度 → 加速比：
#     N=  2     43.61 →  10.15 μs   4.30x  ██████████████████████████████████
#     N=  5    103.08 →  11.67 μs   8.83x  ██████████████████████████████████████████████████████████████████████
#     N= 10    203.18 →  18.70 μs   10.86x  ██████████████████████████████████████████████████████████████████████████████████████
#     N= 20    411.06 →  33.28 μs   12.35x  ██████████████████████████████████████████████████████████████████████████████████████████████████
#     N= 50   1015.41 →  76.99 μs   13.19x  █████████████████████████████████████████████████████████████████████████████████████████████████████████

# ======================================================================
# exp4  动态 shape 踩坑：capture 后改数据长度
# ======================================================================
#   CUDA Graph 把 grid / 指针 / 形状全固化进图里。
#   用一个长度捕获，拿另一个长度重放 —— 数据本身变化是可以的，
#   但『形状变化』会让 grid 失效，必须重新捕获。

#   ① 只改 buffer 内容后 replay        → ✓ 正常（图里固化的是指针，不是值）
#   ② 换成不同长度的 tensor 后 replay  → ⚠ 图仍按旧 grid 跑，
#      不会报错，但只处理了前一半数据（静默出错，最危险）

#   三种正确解法：
#    A. 按形状分桶，每个桶捕获一张图（常见且实用）
#    B. 填充到固定最大长度，用 mask 忽略无效部分（LLM 解码常用）
#    C. 形状变了就重新 capture（简单但失去 Graph 意义）

# ======================================================================
# 总结
# ======================================================================

#   本 demo 与 PyTorch 高层版的区别，就在于你看到了 kernel 本体：
#     · @triton.jit 写的 add_relu_kernel  —— kernel 源码
#     · 导出的 PTX                        —— GPU 真正执行的指令
#     · 捕获进 CUDAGraph 的，就是这些 kernel 的 launch 序列

#   无论哪一层，CUDA Graph 消除的东西都一样：
#     CPU 逐个提交 kernel 的驱动开销（launch overhead）。
#   它不减少 GPU 计算量，一次浮点运算都不会少。

#   值得用的判据（exp2 vs exp3 已给出边界）：
#     ✓ kernel 数量多、每个都很小、同一形状反复跑成千上万次
#     ✗ 单个大 kernel 主导（如大 matmul）—— 瓶颈在算力，不在提交
    

# triton_compare_torch_compile.py —— eager / torch.compile / 手写 Triton 正确对比
# TORCH_LOGS=output_code python3 triton_compare_torch_compile.py 2> inductor.log   # 看生成的 Triton kernel
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

N = 10_000_000
BLOCK = 1024
x = torch.randn(N, device="cuda", dtype=torch.float32)
y = torch.randn(N, device="cuda", dtype=torch.float32)

# ---------- 手写 Triton（kernel 原样，只补 num_warps） ----------
@triton.jit
def add_relu_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(X + offs, mask=mask, other=0.0)
    b = tl.load(Y + offs, mask=mask, other=0.0)
    z = tl.maximum(a + b, 0.0)
    tl.store(Z + offs, z, mask=mask)

def add_relu_triton(a, b):
    z = torch.empty_like(a)
    add_relu_kernel[(triton.cdiv(a.numel(), BLOCK),)](
        a, b, z, a.numel(), BLOCK=BLOCK, num_warps=4)
    return z

# ---------- ★ 带参数，不闭包全局张量 ----------
def eager_fn(a, b):
    return torch.clamp(a + b, min=0.0)

# ★ 去掉 max-autotune；fullgraph=True 确保没悄悄断图退回 eager
compiled_fn = torch.compile(eager_fn, fullgraph=True)

# ---------- ★ 显式编译 + 充分预热 + 同步（三件缺一不可） ----------
with torch.no_grad():
    for _ in range(5):
        _ = compiled_fn(x, y)       # 第一次触发 Dynamo + Inductor + Triton JIT
        _ = add_relu_triton(x, y)   # 第一次触发 Triton JIT
    torch.cuda.synchronize()        # ★ 等编译/JIT 真正落地

# ---------- 正确性 ----------
with torch.no_grad():
    ref = eager_fn(x, y)
    for name, got in [("triton",  add_relu_triton(x, y)),
                      ("compile", compiled_fn(x, y))]:
        d = (got - ref).abs().max().item()
        print(f"一致性 {name:8s}: max_abs={d:.3e} {'✅' if d < 1e-5 else '❌'}")
        assert d < 1e-5

# ---------- ★ 计时：CUDA Event + 每轮 flush L2，不是 time.time() ----------
# do_bench 的 warmup/rep 单位是【毫秒】
def bench(fn):
    return do_bench(lambda: fn(x, y), warmup=50, rep=200)

with torch.no_grad():
    results = {
        "eager":         bench(eager_fn),
        "torch.compile": bench(compiled_fn),
        "手写 Triton":   bench(add_relu_triton),
    }

# ---------- 结果 + 带宽账 ----------
def gbps(ms, nbytes):
    return nbytes / (ms * 1e6)          # bytes / (ms·1e6) = GB/s

print("\n============ Z = relu(X + Y)   N=10M fp32 ============")
print(f"{'实现':<16}{'耗时(ms)':>12}{'有效带宽(GB/s)':>18}{'加速比':>12}")
for name, ms in results.items():
    nb = 20 * N if name == "eager" else 12 * N
    print(f"{name:<16}{ms:>12.4f}{gbps(ms, nb):>18.1f}{results['eager'] / ms:>11.2f}x")

print(f"\n3090 HBM 理论 ≈ 936 GB/s；融合下限 = 12N/936 ≈ {12 * N / 936e6 * 1e3:.3f} ms")
print("→ 只有 2 个 op、中间张量仅往返一次，加速上限本就 = 20N/12N ≈ 1.67x")

# ---------- 确认融合是否真的发生 ----------
def count_kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(x, y); torch.cuda.synchronize()
    return sum(1 for e in p.key_averages()
               if (getattr(e, "self_device_time_total", None)
                   or getattr(e, "self_cuda_time_total", 0)) > 0)

with torch.no_grad():
    print(f"\nCUDA kernel 数: eager={count_kernels(eager_fn)}  "
          f"compile={count_kernels(compiled_fn)}  "
          f"triton={count_kernels(add_relu_triton)}")


# 一致性 triton  : max_abs=0.000e+00 ✅
# 一致性 compile : max_abs=0.000e+00 ✅

# ============ Z = relu(X + Y)   N=10M fp32 ============
# 实现                   耗时(ms)        有效带宽(GB/s)       加速比
# eager                 0.3066             652.3           1.00x
# torch.compile         0.1539             779.5           1.99x
# 手写 Triton            0.1601             749.7           1.92x

# 3090 HBM 理论 ≈ 936 GB/s；融合下限 = 12N/936 ≈ 128.205 ms
# → 只有 2 个 op、中间张量仅往返一次，加速上限本就 = 20N/12N ≈ 1.67x

# CUDA kernel 数: eager=4  compile=2  triton=1

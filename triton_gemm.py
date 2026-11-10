import torch
import triton as tr
import triton.language as tl


def _configs():
    out = []
    for bm, bn, bk, w, s in [
        (128, 256, 64, 8, 3),
        (64,  256, 32, 4, 4),
        (128, 128, 32, 4, 4),
        (128,  64, 32, 4, 3),
        (64,  128, 32, 4, 3),
        (32,   64, 32, 2, 2),
    ]:
        out.append(
            tr.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                num_warps=w,
                num_stages=s,
            )
        )
    return out


@tr.autotune(configs=_configs(), key=["M", "N", "K"])
@tr.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_local = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k_local[None, :] * stride_ak
    b_ptrs = B + offs_k_local[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + offs_k_local
        k_mask = offs_k < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device and a.dtype == b.dtype
    assert a.dtype in (torch.float16, torch.bfloat16)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        tr.cdiv(M, META["BLOCK_M"]) * tr.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # 简单正确性检查：和 torch.matmul 比一下
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    c = matmul(a, b)
    ref = a @ b
    print(f"allclose(atol=1e-2): {torch.allclose(c.float(), ref.float(), atol=1e-2)}")
    print(f"max abs error: {(c.float() - ref.float()).abs().max().item():.4e}")

    # benchmark
    from triton.testing import do_bench
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    for _ in range(3):
        matmul(a, b)
    torch.cuda.synchronize()

    print(f"triton: {do_bench(lambda: matmul(a, b)) * 1000:.0f} us")
    print(f"torch : {do_bench(lambda: a @ b) * 1000:.0f} us")
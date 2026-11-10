import torch
import triton
import triton.language as tl

@triton.jit  # JIT 装饰器，Triton 会将此函数编译为 GPU kernel
def add_kernel(
    x_ptr,      # 输入张量 x 的 GPU 内存指针
    y_ptr,      # 输入张量 y 的 GPU 内存指针
    out_ptr,    # 输出张量的 GPU 内存指针
    M,          # 矩阵行数
    N,          # 矩阵列数（也是行步长/stride）
    BLOCK_M: tl.constexpr,  # 每个 program 在 M 维处理的块大小，constexpr 表示编译期常量
    BLOCK_N: tl.constexpr,  # 每个 program 在 N 维处理的块大小
):
    # 获取当前 program 在 2D grid 中的 ID 
    pid_m = tl.program_id(0)  # 第 0 维（行方向）的 program ID 、对应CUDA 的 blockIdx.y 
    pid_n = tl.program_id(1)  # 第 1 维（列方向）的 program ID 、对应CUDA 的 blockIdx.x

    # 计算当前 program 负责的行索引和列索引范围、tl.arange(0, BLOCK_M) 对应cuda block 内 threadIdx.y
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 形状: [BLOCK_M]、
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 形状: [BLOCK_N]

    # 生成掩码，防止访问越界（当矩阵尺寸不是 BLOCK 整数倍时）
    # rows < M 形状为 [BLOCK_M, 1]，cols < N 形状为 [1, BLOCK_N]，广播后得到 [BLOCK_M, BLOCK_N]
    mask = (rows < M)[:, None] & (cols < N)[None, :]

    # 从全局内存加载数据，按行优先布局计算偏移量: offset = row * N + col
    # rows[:, None] * N + cols[None, :] 生成 [BLOCK_M, BLOCK_N] 的偏移矩阵
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0) # 对应cuda读 x_ptr[idx]
    y = tl.load(y_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0) # 对应cuda读 y_ptr[idx]
 
    # 逐元素相加，并将结果写回全局内存、对应cuda直接写out_ptr[idx]
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_cuda  # Triton kernel 只能在 CUDA 设备上运行

    out = torch.empty_like(x)  # 分配与输入同形状、同 dtype、同设备的输出张量
    M, N = x.shape

    # 计算 2D grid 大小：每个 program 处理 BLOCK_M x BLOCK_N 个元素
    grid = (
        triton.cdiv(M, 128),  # M 维需要的 program 数量（向上取整）
        triton.cdiv(N, 128),  # N 维需要的 program 数量
    )

    # 启动 kernel：grid 决定 launch 多少个 program
    add_kernel[grid](
        x, y, out,
        M, N,
        BLOCK_M=128,  # 行方向块大小
        BLOCK_N=128,  # 列方向块大小
    )
    return out


if __name__ == "__main__":
    M, N = 4, 4  # 小矩阵，方便肉眼核对结果

    # 生成测试数据：x 为 0~15 的序列，y 全 1
    x = torch.arange(M * N, dtype=torch.float32, device="cuda").reshape(M, N)
    y = torch.ones(M, N, device="cuda", dtype=torch.float32)

    print("输入 x:")
    print(x.cpu().numpy())
    print("\n输入 y:")
    print(y.cpu().numpy())

    # Warmup：首次调用会触发 JIT 编译，不计入性能测量
    _ = triton_add(x, y)
    torch.cuda.synchronize()

    out_triton = triton_add(x, y)

    print("\nTriton 输出:")
    print(out_triton.cpu().numpy())

    # 用 PyTorch 原生加法作为正确性参考
    out_torch = x + y
    print("\nTorch 参考输出:")
    print(out_torch.cpu().numpy())

    # 验证 Triton 结果与 PyTorch 结果是否一致
    err = (out_triton - out_torch).abs().max().item()
    print(f"\nMax error: {err:.6f}")
    print("Check:", "PASS" if err < 1e-4 else "FAIL")
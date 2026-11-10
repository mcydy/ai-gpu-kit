import torch
import triton
import triton.language as tl

@triton.jit
def bilinear_kernel(
    src_ptr, dst_ptr,
    src_h, src_w,
    dst_h, dst_w,
    scale_h, scale_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (dst_h * dst_w)

    dst_y = offsets // dst_w
    dst_x = offsets % dst_w

    src_x = (dst_x.to(tl.float32) + 0.5) / scale_w - 0.5
    src_y = (dst_y.to(tl.float32) + 0.5) / scale_h - 0.5

    src_x = tl.clamp(src_x, 0.0, src_w.to(tl.float32) - 1.0 - 1e-6)
    src_y = tl.clamp(src_y, 0.0, src_h.to(tl.float32) - 1.0 - 1e-6)

    x0 = tl.floor(src_x).to(tl.int64)
    y0 = tl.floor(src_y).to(tl.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = src_x - x0.to(tl.float32)
    dy = src_y - y0.to(tl.float32)

    base00 = y0 * src_w + x0
    base01 = y0 * src_w + x1
    base10 = y1 * src_w + x0
    base11 = y1 * src_w + x1

    v00 = tl.load(src_ptr + base00, mask=mask, other=0.0)
    v01 = tl.load(src_ptr + base01, mask=mask, other=0.0)
    v10 = tl.load(src_ptr + base10, mask=mask, other=0.0)
    v11 = tl.load(src_ptr + base11, mask=mask, other=0.0)

    v0 = v00 * (1.0 - dx) + v01 * dx
    v1 = v10 * (1.0 - dx) + v11 * dx
    val = v0 * (1.0 - dy) + v1 * dy

    tl.store(dst_ptr + offsets, val, mask=mask)

def bilinear_triton(src: torch.Tensor, scale: float) -> torch.Tensor:
    src_h, src_w = src.shape  #src_h, src_w = src.shape
    dst_h, dst_w = int(src_h * scale), int(src_w * scale)  # 根据缩放比例计算目标尺寸
    dst = torch.empty(dst_h, dst_w, device=src.device, dtype=torch.float32) #分配输出张量，数据类型固定为 float32
    
    BLOCK = 256 # 每个 block 处理 256 个输出像素
    grid = (triton.cdiv(dst_h * dst_w, BLOCK),)  # 计算 grid：把 dst_h * dst_w 个输出像素按 256 个一组划分。 
    bilinear_kernel[grid]( #启动 Triton kernel。如果目标尺寸1024*1024 那么grid = (4056,)，即启动 4056 个 program（block），每个 block 处理 256 个像素。
        src, dst, src_h, src_w, dst_h, dst_w, scale, scale, BLOCK=BLOCK
    )
    return dst


def bilinear_cpu(src, scale):
    """修正版：用 floor 而不是 // 1"""
    src_h, src_w = src.shape
    dst_h, dst_w = int(src_h * scale), int(src_w * scale)
    dst = torch.empty(dst_h, dst_w)
    for dy in range(dst_h):
        sy = (dy + 0.5) / scale - 0.5
        y0 = int(sy // 1) if sy >= 0 else int(sy - 0.99999)  # floor
        y1 = min(y0 + 1, src_h - 1)
        fy = sy - y0
        for dx in range(dst_w):
            sx = (dx + 0.5) / scale - 0.5
            x0 = int(sx // 1) if sx >= 0 else int(sx - 0.99999)
            x1 = min(x0 + 1, src_w - 1)
            fx = sx - x0
            v00, v01 = src[y0, x0], src[y0, x1]
            v10, v11 = src[y1, x0], src[y1, x1]
            v0 = v00*(1-fx) + v01*fx
            v1 = v10*(1-fx) + v11*fx
            dst[dy, dx] = v0*(1-fy) + v1*fy
    return dst


if __name__ == "__main__":
    scale = 2.0
    src = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)

    print("源图:")
    print(src.cpu().numpy())

    # Warmup（消除 JIT 编译时间）
    _ = bilinear_triton(src, scale)
    torch.cuda.synchronize()

    # GPU 计时（warmup 后）
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    dst_triton = bilinear_triton(src, scale)
    t1.record()
    torch.cuda.synchronize()
    gpu_ms = t0.elapsed_time(t1)

    # CPU（用 numpy 做正确参考，不依赖手写 CPU 实现）
    import numpy as np
    src_np = src.cpu().numpy()
    dst_h, dst_w = 8, 8
    dst_cpu = np.zeros((dst_h, dst_w), dtype=np.float32)
    for dy in range(dst_h):
        sy = (dy + 0.5) / scale - 0.5
        sy = np.clip(sy, 0, 3)
        y0 = int(np.floor(sy)); y1 = min(y0 + 1, 3)
        fy = sy - y0
        for dx in range(dst_w):
            sx = (dx + 0.5) / scale - 0.5
            sx = np.clip(sx, 0, 3)
            x0 = int(np.floor(sx)); x1 = min(x0 + 1, 3)
            fx = sx - x0
            v00, v01 = src_np[y0, x0], src_np[y0, x1]
            v10, v11 = src_np[y1, x0], src_np[y1, x1]
            dst_cpu[dy, dx] = (v00*(1-fx)+v01*fx)*(1-fy) + (v10*(1-fx)+v11*fx)*fy

    print("\nTriton 结果:")
    print(dst_triton.cpu().numpy())
    print("\nNumPy 参考:")
    print(dst_cpu)

    max_err = (dst_triton.cpu().numpy() - dst_cpu).max()
    print(f"\nGPU(Triton)耗时: {gpu_ms:.4f} ms")
    print(f"最大误差:       {max_err:.6f}")
    print("验证:", " PASS" if max_err < 1e-4 else " FAIL")
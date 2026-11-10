import torch, time

N = 8000
device = "cuda"  

# 直观看到 GPU 快多少
for dev in ["cpu", "cuda"]:
    a = torch.randn(N, N, device=dev)
    b = torch.randn(N, N, device=dev)
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    # 矩阵乘法
    c = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"{dev:5s}: {time.time()-t0:.3f}s  shape={c.shape}")

# 显存占用
print("显存已用:", round(torch.cuda.memory_allocated()/1024**3, 2), "GB")
print("显存峰值:", round(torch.cuda.max_memory_allocated()/1024**3, 2), "GB")

import torch
print("设备数:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name}  显存={p.total_memory/1024**3:.1f}GB  "
          f"算力={p.major}.{p.minor}")

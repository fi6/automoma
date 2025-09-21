import torch
import time

# export CUDA_VISIBLE_DEVICES=${gpu_id}
# Allocate about 10000MB (10GB) on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 10000 * 1024 * 1024 // 4  # float32 uses 4 bytes
x = torch.empty(size, dtype=torch.float32, device=device)

# Keep the GPU busy for 1 minute
start = time.time()
while time.time() - start < 60:
    x = x * 1.000001  # simple operation to keep GPU active
    torch.cuda.synchronize()

print("Test finished.")
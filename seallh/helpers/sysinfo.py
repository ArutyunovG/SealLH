import torch

def get_allocated_gpu_mem_mb():
    return torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0

def get_allocated_gpu_mem_gb():
    return torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0


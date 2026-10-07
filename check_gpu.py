"""Mostra a GPU em uso. Rode dentro do terminal do VSCode e tire o print (item 7)."""
import subprocess, torch

print("PyTorch:", torch.__version__, "| CUDA do PyTorch:", torch.version.cuda)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1024**3:.1f} GB | Compute capability: {p.major}.{p.minor}")
    print("bf16 suportado:", torch.cuda.is_bf16_supported())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


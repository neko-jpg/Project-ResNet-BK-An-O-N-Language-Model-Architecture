import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA device available")

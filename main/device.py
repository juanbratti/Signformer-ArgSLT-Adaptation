import torch
import os

# Clear any existing device settings
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("HIP_VISIBLE_DEVICES", None)

# Set ROCm-specific environment variables
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA to prevent memory issues

print("Environment variables:")
print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES"))

print("\nPyTorch configuration:")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Current CUDA device:", torch.cuda.current_device())
    
    # Initialize CUDA
    torch.cuda.init()
    
    # Try to create a tensor on GPU to verify it works
    try:
        with torch.cuda.device(0):
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            print("Successfully created tensor on GPU:", x)
            print("Tensor device:", x.device)
    except Exception as e:
        print("Error creating tensor on GPU:", e)

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nUsing GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("\nUsing CPU")
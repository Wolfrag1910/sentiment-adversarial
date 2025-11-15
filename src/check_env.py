import sys
import torch
import numpy as np

print("Python:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("NumPy version:", np.__version__)
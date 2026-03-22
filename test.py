import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

x = torch.rand(3, 3).to(device)
print(x)
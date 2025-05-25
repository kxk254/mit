import torch
print(torch.cuda.is_available())  # should be True
print(torch.version.cuda)         # should return your CUDA version
print(torch.cuda.get_device_name(0))  # should return your GPU name
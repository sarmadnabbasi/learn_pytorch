import torch.cuda

print(f"{torch.cuda.get_device_name(0)}\n\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda0 = torch.device('cuda:0')

torch.cuda.get_device_name(0)

tensor_A = torch.tensor([])
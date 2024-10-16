#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lesson", help="Lesson number from youtube video", type= int, default=0)
parser.add_argument("-v","--verbosity", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
lesson_number = args.lesson

import torch
x = torch.rand(5, 3)
print(torch.cuda.get_device_name(0))

###### Utilities ######
## Setup device ##
cuda0 = torch.device('cuda:0')


###### 18. Tensor Attributes ######
def lesson_18():
      print("\n###### 18. Tensor Attributes ######")
      int_32_tensor= torch.tensor ([3, 6, 9], dtype=torch.int32)
      float_32_tensor = torch.tensor([3, 6, 9], dtype=torch.float32)
      print("float_32_tensor:",float_32_tensor)
      print("float_32_tensor * int_32_tensor", float_32_tensor*int_32_tensor)

      print("\n### Getting information from tensors (tensor attributes")
      print("Tensors not right datatype - to get the datatype from a tensor, can use tensor.dtype\n"
            "Tensors not right shape - to get shape from a tensor, can use tensor.shape\n"
            "Tensors not on the right device - to get device from a tensor, can use tensor.device\n")

      some_tensor = torch.rand(3, 4)
      print("some tensor: ", some_tensor)
      print(f"Datatype of tensor: {some_tensor.dtype}")
      print(f"Shape of tensor: {some_tensor.shape}")
      print(f"Device tensor is on: {some_tensor.device}")

      print("\nChange Device")
      some_tensor = some_tensor.to(cuda0)
      print(f"some_tensor device: {some_tensor.device}")
      print("__________________________________")


###### 19. Manipulating Tensors (tensor operations ######
def lesson_19 ():
      print("\n###### 19. Manipulating Tensors (tensor operations ######")
      print("Tensor operations include:\n"
            "-  Addition\n"
            "-  Subtraction\n"
            "-  Multiplication\n"
            "-  Division\n"
            "-  Matrix Mult\n")

      tensor = torch.tensor([1, 2, 3])
      print(tensor)
      print(f"Tensor addition: {tensor + 100}")
      print(f"Tensor Multiplication: {tensor * 100}")
      print(f"Tensor Multiplication: {tensor * 100}")
      print(f"or Tensor Multiplication: {torch.mul(tensor, 100)}")

match

print(None)
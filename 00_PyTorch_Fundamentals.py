#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

#### Imports
import argparse
from time import process_time, process_time_ns
import time
import torch
import numpy as np
#####################################

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lesson", help="Lesson number from youtube video", type= int, default=29)
parser.add_argument("-v","--verbosity", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
lesson_number = args.lesson
#######################################


###### Utilities ######
## Setup device ##
print(f"{torch.cuda.get_device_name(0)}\n\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda0 = torch.device('cuda:0')
#######################################

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
      print("__________________________________\n\n")


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
      print("__________________________________\n\n")

###### 20. Matrix Multiplication ######
def lesson_20 ():
      print("\n###### 20. Matrix Multiplication ######")
      print("### Element wise Multiplication")
      tensor = torch.tensor([11,12,13], dtype=torch.int32, device=cuda0)
      print(f"tensor * tensor = {tensor * tensor}")

      print("\n### Matrix transpose (for shape fixing) -- tensor.T")
      print(f"tensor: {tensor}")
      tensor = torch.tensor([[1, 2, 3],
                             [4, 5, 6]], dtype=torch.float32
                            )
      print(f"tensor : {tensor}, shape : {tensor.shape}")
      print(f"tensor.T : {tensor.T}, shape : {tensor.T.shape}")

      print("\n### Dot product -- torch.matmul or torch.mm")
      tensor_A = torch.tensor([[1, 2, 3],
                             [4, 5, 6]], dtype=torch.float32
                            )
      tensor_B = torch.tensor([[7, 8, 9],
                             [10, 11, 12]], dtype=torch.float32
                            )

      print(f"Original shape: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
      print(f"change tensor.B to tensor_B.T as inner dimensions must be same")
      print(f"Multiplying tensor_A and tensor_B.T: {tensor_A.shape} and {tensor_B.shape}")
      print(f"torch.matmul(tensor_A,tensor_B.T)) = {torch.matmul(tensor_A,tensor_B.T)}")

      print("__________________________________\n\n")


###### 23. Finding min, max, mean and sum ######
def lesson_23 ():
      x = torch.arange(0, 100, 10, dtype=torch.float32)
      print(f"tensor: {x}")

      print("\n### Find min -- torch.min(tensor) or tensor.min()")
      print(f"x.min(): {x.min()}")

      print("\n### Find max -- torch.max(tensor) or tensor.max()")
      print(f"x.max(): {x.max()}")

      print("\n### Find mean -- torch.mean(tensor) or tensor.mean()")
      print("Note: Change dtype to Float or Long")
      print(f"x.type(dtype=torch.float32).mean(): {x.type(dtype=torch.float32).mean()}")

      print("\n### Find sum-- torch.sum(tensor) or tensor.sum()")
      print(f"x.sum(): {x.sum()}")

      print("\n### Find positional min and max-- torch.argmin(tensor) or tensor.argmin() -- same for argmax")
      print(f"x.argmin(): {x.argmin()}, x.argmax(): {x.argmax()}")

      print("__________________________________\n\n")

###### 25. Reshaping, viewing and stacking (stacking, squeezing and unsqueezing tensors ######
def lesson_25 ():
      print("* Reshaping - Reshapes an inout tensor to a defined shape\n"
            "* View - Return a view of an input tensor of certain shape but keep the same memory as the original\n"
            "* Stacking - combine multiple tensors on top of each other (vstack) or side-by-side (hstack)\n"
            "* Squeeze - removes all '1' dimensions from a tensor\n"
            "* Unqueeze - add a '1' dimention to a target tensor\n"
            "* Permute - Return a view of the input dimension permuted (swapped) in a certain way.")

      x = torch.arange(1., 10.)
      print(f"tensor: {x}, shape: {x.shape}")
      print("\n### Reshape Tensor -- tensor.reshape()")
      print(f"tensor.reshape(9,1) : {x.reshape(9,1)}")

      print("\n### View -- tensor.view()")
      print("Note: the new tensor will share the same memory so changing the new will change the previous")
      print(f"x: {x}")
      z = x.view(1,9)
      print(f"z = x.view(1,9): {z}")
      print(f"z: {z}, shape: {z.shape}")

      print("Changing z> z[:, 0] = 5)")
      z[:, 0] = 5
      print(f"x: {x}")

      print("\n### Stack -- tensor.stack()")
      print(f"x: {x}")
      x_stacked = torch.stack([x,x,x,x,x], dim=0)
      print(f"Stacked 5 times\ntorch.stack([x,x,x,x,x], dim=0): {x_stacked}")

      print("__________________________________\n\n")

##### 26. Squeezing, unsqueezing and permuting tensors
def lesson_26():
      x = torch.arange(1., 10)
      print(f"tensor: {x}, shape: {x.shape}")
      x = x.reshape((1,-1))
      print(f"tensor.reshape(1,-1): {x}, shape: {x.shape}")
      print("\n### Squeeze -- tensor.squeeze() -- remove extra '1' dimension")
      print(f"tensor.squeeze(): {x.squeeze()}, shape: {x.squeeze().shape}")

      x = x.squeeze()
      print("\n### Unsqueeze -- tensor.unsqueeze(dim=) -- add extra '1' dimension")
      print(f"tensor.unsqueeze(): {x.unsqueeze(dim=1)}, shape: {x.unsqueeze(dim=1).shape}")

      print("\n### Permute -- tensor.permute() -- rearranges the dimensions of tensor in a specific order and gives view")
      x_original = torch.rand(size=(224, 224, 3)) #random for image data [height, width, channel]
      print(f"example image tensor shape: {x_original.shape}")
      x_permuted = x_original.permute(2,0,1)
      print(f"swap color channel to 1st dimension> tensor.permute(2, 0, 1).shape: {x_permuted.shape}")


      print("__________________________________\n\n")

##### 27. Selecting data (indexing)
def lesson_27():
      x = torch.arange(1, 10).reshape(1,3,3)
      print(f"tensor: {x}, shape: {x.shape}")
      print(f"tensor[0]: {x[0]}")
      print(f"tensor[:,:,0]: {x[:,:,0]} >> : is used for select all")
      print("__________________________________\n\n")

##### 28. Pytorch and Numpy #####
def lesson_28():
      print("Data in NumPy, want in PyTorch tensor -> torch.from_numpy(ndarray)\n"
            "PyTorch tensor -> NumPy -> torch.Tensor.numpy()")
      print("\n### NumPy array to tensor")
      array = np.arange(1., 10.)
      print(f"NumPy array: np.arange(1., 10.) = {array}")
      tensor = torch.from_numpy(array)
      print(f"to Tensor: torch.from_numpy(array) = {tensor}")
      print(f"to NumPy: tensor.numpy() = {tensor.numpy()}")
      print("Note: NumPy detault datatype is float64 and PyTorch's default datatype is float32")
      print("__________________________________\n\n")

##### 29. PyTorch reproducibility (taking the random out of random) #####
def lesson_29():
      print("##### 29. PyTorch reproducibility (taking the random out of random) #####\n")
      print("To reduce the randomness in neural networks and pytorch comes the concept of a random seed")

      random_tensor_A = torch.rand(3, 4)
      random_tensor_B = torch.rand(3, 4)
      print(f"random_tensor_A = torch.rand(3,4): {random_tensor_A}")
      print(f"random_tensor_B = torch.rand(3,4) {random_tensor_B}")
      print(f"\nCheck if equal> random_tensor_A == random_tensor_B: {random_tensor_A == random_tensor_B} ")

      RANDOM_SEED = 42
      print(f"\nRANDOM_SEED: {RANDOM_SEED}")

      print(f"\nSet random seed> torch.manual_seed(RANDOM_SEED)")
      torch.manual_seed(RANDOM_SEED)
      random_tensor_C = torch.rand(3, 4)
      print(f"random_tensor_C = torch.rand(3,4): {random_tensor_C}")

      print(f"\nSet random seed> torch.manual_seed(RANDOM_SEED)")
      torch.manual_seed(RANDOM_SEED)
      random_tensor_D = torch.rand(3, 4)
      print(f"random_tensor_D = torch.rand(3,4): {random_tensor_D}")

      print(f"\nCheck if equal> random_tensor_C == random_tensor_D: {random_tensor_C == random_tensor_D} ")

      print("Note: You have to call torch.manual_seed before every rand tensor initialization")
      print("__________________________________\n\n")

if __name__ == "__main__":
      func = "lesson_"+str(args.lesson)

      exec(f"x = {func}")

      t = process_time_ns()
      start = time.time_ns()
      x()
      print("\n*** Time taken ***")
      print(f"Elapsed CPU time: {(process_time_ns() - t)/(1000*1000)}ms")
      print(f"Elapsed Wall time: {(time.time_ns()-start)/(1000*1000)}ms")
      print("********************\n")

      #print(None)

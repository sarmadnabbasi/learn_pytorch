#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

#### Imports
import argparse
from time import process_time, process_time_ns
import time
import torch
from matplotlib.pyplot import legend
from networkx.algorithms.bipartite import color
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#####################################

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lesson", help="Lesson number from youtube video", type= int, default=48)
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

###### 35. Creating a dataset with linear regression ######
def lesson_35():
      print("\n###### 35. Creating a dataset with linear regression  ######")
      weight = 0.7
      bias = 0.3
      start = 0
      end = 1
      step = 0.02
      X = torch.arange(start, end, step)
      y = bias + weight*X
      print("\nKnown parameters")
      print(f"weight: {weight}")
      print(f"bias: {bias}")

      print(f"\nX = torch.arange(start, end, step) = {X}")
      print(f"y = bias + weight*X = {y}")

      return X, y

      print("__________________________________\n\n")

###### 36. Creating training and test sets (the most important concept in ML) ######
def lesson_36():
      print("\n###### 36. Creating training and test sets (the most important concept in ML) ######")
      X, y = lesson_35()

      print("\n\nCreate Test/Train split")
      train_split = int(0.8 * len(X))

      X_train, y_train  = X[:train_split], y[:train_split]
      X_test, y_test  = X[train_split:], y[train_split:]
      print("80% train data > train_split = int(0.8 * len(X))")
      print("X_train, y_train  = X[:train_split], y[:train_split]")
      print("X_test, y_test  = X[train_split:], y[:train_split:]")
      print(f"length of X_train: {X_train.shape}, y_train: {y_train.shape}\n"
            f"length of X_test: {X_test.shape}, y_test: {y_test.shape}")

      print("\nPlotting Data")
      plot_predictions(X_train, y_train, X_test, y_test)

      return X_train, y_train, X_test, y_test
      print("__________________________________\n\n")

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions = None):

      plt.figure(figsize = (10, 7))

      plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
      plt.scatter(test_data, test_labels, c="g", s=5, label="Test data")

      if predictions is not None:
            plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

      plt.legend(prop={"size": 14})
      plt.show()

###### 38. Crating our first PyTorch model ######
def lesson_38():
      print("\n###### 38. Crating our first PyTorch model ######")
      print("\n## Create linear regression model class")

      print("__________________________________\n\n")

class LinearRegressionModel(nn.Module):
      def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.rand(1,
                                                   requires_grad=True,
                                                   dtype=torch.float32))
            self.bias = nn.Parameter(torch.rand(1,
                                                   requires_grad=True,
                                                   dtype=torch.float32))

            # Forward method to define the computation in the model
      def forward(self, x: torch.Tensor) -> torch.tensor:
                  return self.weights*x +self.bias #Linear regression formula



###### 40. Discssing important model building classes ######
def lesson_40():
      print("\n###### 40. Discssing important model building classes ######")
      print("\n* torch.nn - Contains all of the builidngs block for computational graphs (a neural network)\n"
            "* torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us\n"
            "* torch.nn.Module - The base class for all the neural network modules, if subclass, you should overwrite forward()\n"
            "* torch.optim - Optimized in PyTorch. Will help with gradient decent\n"
            "* def forward(), this method define what happens in forward computation (forward propagation)\n"
            "https://pytorch.org/tutorials/beginner/ptcheat.html"
            "")

      print("__________________________________\n\n")

###### 41. Checking the content of our PyTorch Model ######
def lesson_41():
      print("\n###### 41. Checking the content of our PyTorch Model ######")
      print("\nCheck what is inside the model..")
      RANDOM_SEED= 42
      torch.manual_seed(RANDOM_SEED)
      print(f"create random seed 42 > torch.manual_seed(RANDOM_SEED)\n")

      print(f"create Model > model_0 = LinearRegressionModel()\n")
      model_0 = LinearRegressionModel()

      print(f"list parameters of the model_0 > list(model_0.parameters()) : {list(model_0.parameters())}")
      print(f"\nlist named of the model_0 > model_0.state_dict() : {model_0.state_dict()}")

      return model_0
      print("__________________________________\n\n")

###### 42. Making Prediction with our Model ######
def lesson_42():

      X_train, y_train, X_test, y_test = lesson_36()
      model_0 = lesson_41()

      print("\n###### 42. Making Prediction with our Model ######")
      print("\nTo check predictive power of our model > predict y_test based on X_test\n")
      print("\nRun through forward()")
      print("\nuse torch.inference_model() to disable the gradients for predictions, save resources")
      with torch.inference_mode():
            y_preds = model_0.forward(X_test)

      plot_predictions(X_train, y_train, X_test, y_test, predictions = y_preds)
      print("__________________________________\n\n")

###### 43. Training a model with PyTorch (intuition building) ######
def lesson_43():
      print("\n###### 43. Training a model with PyTorch (intuition building) ######")
      print("\n## Make Loss function (other names cost, criterion function) ")
      print("\n* Loss function > Measures how wrong our preconditions are")
      print("\n* Optimizer > Updates parameters (weights and bias) depending on loss function")

      print("\n## Required for PyTorch\n"
            "* A training loop\n"
            "* A testing loop")

      model_0 = LinearRegressionModel()

      return model_0

      print("__________________________________\n\n")


###### 44. Setting up a loss function and optimizer ######
def lesson_44():
      print("\n###### 44. Setting up a loss function and optimizer ######")

      model_0 = LinearRegressionModel()
      print("\n## Setup a Loss function > (MAE, L1Norm) > nn.L1Loss()")
      loss_fn = nn.L1Loss()
      print("\n## Setup an optimizer > (SGD) > torch.optim.SDG(params=model_0.parameters(), lr=0.01)")
      optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

      model_0 = LinearRegressionModel()

      return model_0

      print("__________________________________\n\n")


###### 45-49. PyTorch training and testing loop  ######
def lesson_45():
      print("\n###### 45-49. PyTorch training and testing loops ######")
      print("\nLoad data set > X_train, y_train, X_test, y_test = lesson_36()")
      X_train, y_train, X_test, y_test = lesson_36()

      print("\n## Things we need in training loop\n"
            "0. Loop through the data\n"
            "1. Forward pass > forward()\n"
            "2. Calculate loss > loss_fn\n"
            "3. Optimizer zero grad\n"
            "4. Loss backward> Backward propagation to calculate the gradients of each of the parameters of our model with respoce to the loss\n"
            "5. Optimizer loss> use the optimizer to adjust our model's parameters to try and improve the loss (Gradient descent)")

      RANDOM_SEED = 42
      torch.manual_seed(RANDOM_SEED)
      model_0 = LinearRegressionModel()

      print(f"\ninitial model parameters: {model_0.state_dict()}")
      with torch.inference_mode():
            plot_predictions(X_train, y_train, X_test, y_test, model_0(X_test))



      print("\n## epochs > 1 loop through the data > epochs (Hyperparameter")
      epochs = 100
      print("\n## Setup a Loss function > (MAE, L1Norm) > nn.L1Loss()")
      loss_fn = nn.L1Loss()
      print("\n## Setup an optimizer > (SGD) > torch.optim.SDG(params=model_0.parameters(), lr=0.001)")
      optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

      print("\n0. Loop through the data\n")
      print("\nSet model in training mode > model_0.train() > set all parameters that require gradients to require grad")
      print("\n1. Forward Pass")
      print("y_pred = model_0.forward(X_train)")
      print("\n2. Calculate Loss")
      print("loss = loss_fn(y_pred, y_train)")
      print("\n3. optimizer zero grad")
      print("\n4. Back Propagation")
      print("\n5. Optimizer Step (gradient decent")
      print("Set model in eval mode > model_0.eval() > turn off grad")

      loss_save = []
      loss_test_save =[]
      epoch_save =[]
      for epoch in range(epochs):

            model_0.train()

            y_pred = model_0.forward(X_train)

            loss = loss_fn(y_pred, y_train)
            #print(f"Loss : {loss}")

            loss_save.append(loss.item())
            epoch_save.append(epoch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            model_0.eval()


            with torch.inference_mode():
                  loss_test = loss_fn(model_0.forward(X_test), y_test)
                  loss_test_save.append((loss_test.item()))

            if epoch % 10 == 0:
                  print(f"model_0.state_dict() : {model_0.state_dict()}")
                  print(f"Epoch: {epoch} | training_loss: {loss} | test_loss: {loss_test} ")


      with torch.inference_mode():
            plot_predictions(X_train, y_train, X_test, y_test, model_0(X_test))

      # Plot the loss curves
      plt.plot(epoch_save, loss_save, color = 'b', label = "Training Loss")
      plt.plot(epoch_save, loss_test_save, color = 'r', label= "Test Loss")
      plt.title("Training and test loss curves")
      plt.ylabel("Loss")
      plt.xlabel("Epochs")
      plt.legend()
      plt.show()
      print("__________________________________\n\n")

      return model_0

lesson_48 = lesson_49 =lesson_45

###### 51. Saving/loading a model ######
def lesson_51():
      print("###### 51. Saving/loading a model ######")
      print("There are three main methods you should know about for saving and loading models")
      print("1. 'torch.save()' - Saves model in Python Pickle format" )
      print("2. 'torch.load()' - Allows you to load a saved PyTorch object")
      print("3. 'torch.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary")

      print("\n## Saving our PyTorch Model")

      print("1. Create Models directiory")
      MODEL_PATH = Path("models")
      MODEL_PATH.mkdir(parents = True, exist_ok = True)





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

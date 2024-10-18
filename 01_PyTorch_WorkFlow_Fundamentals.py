#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

#### Imports
import argparse
from time import process_time, process_time_ns
import time
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
#####################################

#### Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lesson", help="Lesson number from youtube video", type= int, default=36)
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



class LinearRegression(nn.Module):
      def __init__(self):
            super.__init__()
            self.weights = nn.Parameter
            self.bias = nn.Parameter
            start=None




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

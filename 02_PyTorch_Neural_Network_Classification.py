import torch
from torch import nn
import argparse
import time
import sklearn

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd



# Make 1000 samples
n_samples = 1000

# Create circles


print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    device = "cpu"
    print(f"Device name: {device}")


parser = argparse.ArgumentParser()
parser.add_argument("--lesson", help="Lesson number from youtube video", type=int, default=62)

args = parser.parse_args()
lesson_number = args.lesson

###### 62, 64, 66. Architecture of a classification Neural Network ######
def lesson_62():
    print("###### 62. Architecture of a classification Neural Network ######\n")
    print("###### 64. Turning the data into tensors ######\n")
    print("###### 66. Coding a Neural Network for classification  data ######\n")
    print("###### 69. Loss, Optimizer, and Evaluation ######\n")

    X, y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)

    len(X), len(y)

    print(f"First 5 samples of X:\n {X[:5]}")
    print(f"First 5 samples of y:\n {y[:5]}")

    circles = pd.DataFrame({"X1": X[:, 0],
                            "X2": X[:, 1],
                            "label": y})

    circles.head(10)

    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu)

    print("\n## Check Input and Output shapes")
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    X_sample = X[0]
    y_sample = y[0]
    print(f"Sample 0 of X[0] = {X_sample}, y[0] = {y_sample}")
    print(f"Shape of sample 0 of X[0] = {X_sample.shape}, y[0] = {y_sample.shape}")

    print("###### 64 Turning the data into tensors ######\n")
    print("\n## Change to tensors")
    X = torch.from_numpy(X).type(dtype=torch.float32)
    y = torch.from_numpy(y).type(dtype=torch.float32)

    print(f"5 samples > torch.from_numpy(X) ={X[:5]}, torch.from_numpy(y) ={y[:5]}")

    print("\n## Train test split")

    print("Using Sklearn > X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Shape of > X_train = {X_train.shape}, X_test = {X_test.shape}, y_train = {y_train.shape}, y_test = {y_test.shape}")

    print("###### 66. Coding a Neural Network for classification  data ######\n")

    model_0 = CircleModelV0().to(device=device)

    model_1 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1)
    ).to(device)

    with torch.inference_mode():
        untrained_preds = model_1(X_test.to(device))
        print(f"len(X_test): {len(X_test)}, X_test.shape: {X_test.shape}")
        print(f"len(untrained_preds): {len(untrained_preds)}, untrained_preds.shape: {untrained_preds.shape}")
        print(f"First 10 preds  > untrained_preds[:10]): {untrained_preds[:10]}")
        print(f"First 10 labels > y_test[:10]): {y_test[:10]}")



    ## Hyperparameters
    epochs = 1000
    lr = 0.01

    loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(params=model_1.parameters(), lr=lr)
    for epoch in range(epochs):

        model_0.train()

    print("__________________________________\n\n")


    print(None)



class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
        return self.two_linear_layers(x)



#########################
def main():
    func = "lesson_"+str(lesson_number)
    exec(f"x = {func}")

    start_time_CPU = time.process_time_ns()
    start_time_wall = time.time_ns()
    lesson_62()
    end_time_CPU = time.process_time_ns()
    end_time_wall = time.time_ns()

    print(f"Elapsed CPU time: {end_time_CPU-start_time_CPU}")
    print(f"Elapsed Wall time: {start_time_wall - end_time_wall}")


if __name__ == "__main__":
    func = "lesson_" + str(lesson_number)
    exec(f"x = {func}")

    start_time_CPU = time.process_time_ns()
    start_time_wall = time.time_ns()
    x()
    end_time_CPU = time.process_time_ns()
    end_time_wall = time.time_ns()

    print(f"Elapsed CPU time: {end_time_CPU - start_time_CPU}")
    print(f"Elapsed Wall time: {start_time_wall - end_time_wall}")
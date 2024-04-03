import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_sigmoid_values(data):
    print("Input Value\tSigmoid")
    print("-------------------------")
    for val in data:
        sigmoid_val = sigmoid(val)
        print(f"{val:.2f}\t\t{sigmoid_val:.6f}")

if __name__ == "__main__":
    random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
    print_sigmoid_values(random_values)

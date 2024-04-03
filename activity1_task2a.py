import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


## modified code
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

def apply_activation_functions(data):
    print("Input Value\tReLU\t\tLeaky ReLU\tTanh")
    print("--------------------------------------------")
    for val in data:
        relu_val = relu(val)
        leaky_relu_val = leaky_relu(val)
        tanh_val = tanh(val)
        print(f"{val:.2f}\t\t{relu_val:.6f}\t{leaky_relu_val:.6f}\t{tanh_val:.6f}")
## same as before
if __name__ == "__main__":
    random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
    apply_activation_functions(random_values)

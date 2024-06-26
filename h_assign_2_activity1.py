import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-5, 5, 100)

# Calculate activation function values
sigmoid_y = sigmoid(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_y, label='Sigmoid', color='blue')
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, relu_y, label='ReLU', color='orange')
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU')
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, tanh_y, label='Tanh', color='red')
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

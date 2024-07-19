# Feedforward Neural Network for Fashion-MNIST Classification

## Introduction

This project implements a feedforward neural network (FNN) from scratch using numpy for the classification of images in the Fashion-MNIST dataset. The goal is to develop a neural network capable of classifying images into one of ten classes, such as T-shirt, trouser, pullover, dress, and more. This project also includes the implementation of backpropagation and various optimization techniques without using any automatic differentiation packages.

## Problem Statement

The objective is to build, train, and evaluate a feedforward neural network for classifying Fashion-MNIST images. The network will take an input image of 28x28 pixels and output a probability distribution over 10 classes. The implementation must include the flexibility to adjust the number of hidden layers, the number of neurons in each hidden layer, and support for multiple optimization algorithms.

## Methodology

The project follows these key steps:

1. **Data Loading and Visualization**: Download and visualize the Fashion-MNIST dataset.
2. **Feedforward Neural Network Implementation**: Implement a flexible neural network using numpy.
3. **Backpropagation and Optimization**: Implement the backpropagation algorithm with support for SGD, Momentum-based GD, and Nesterov Accelerated Gradient Descent.
4. **Hyperparameter Tuning**: Train and validate the model using different hyperparameter configurations to find the best performing model.
5. **Model Evaluation**: Evaluate the best model on the test set and report the accuracy and confusion matrix.

## Implementation Details

### Data Loading and Visualization

The Fashion-MNIST dataset is downloaded using the `torchvision.datasets` library and converted into numpy arrays. A sample image from each class is plotted in a grid.

```python
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Download and load the Fashion-MNIST dataset
train_set = datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

# Convert to numpy arrays
X_train = train_set.data.numpy()
y_train = train_set.targets.numpy()
X_test = test_set.data.numpy()
y_test = test_set.targets.numpy()

# Plot sample images
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i in range(10):
    sample_idx = np.where(y_train == i)[0][0]
    axes[i].imshow(X_train[sample_idx], cmap='gray')
    axes[i].set_title(f'Class {i}')
    axes[i].axis('off')
plt.show()

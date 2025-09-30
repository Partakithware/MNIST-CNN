MNIST CNN from Scratch in NumPy

A minimal Convolutional Neural Network (CNN) implementation for MNIST handwritten digit classification, built entirely from scratch using NumPy, without frameworks like TensorFlow or PyTorch.

This project demonstrates how convolution, max-pooling, fully connected layers, and backpropagation work at a low level. It achieves near-perfect accuracy on MNIST, showing that even a small network can be highly effective when implemented correctly.

Features

Convolutional layer with im2col optimization.

2x2 max-pooling with backward pass support.

Fully-connected hidden and output layers.

Dual optimizer updates: Adam + SGD with momentum.

Training from scratch on full MNIST dataset.

Logs predictions vs actual labels for test set.

Results

Achieved ~98% test accuracy after 5 epochs.
Full MNIST test accuracy: 97.96%


Sample predictions for first 10 images:
Predictions: [7 2 1 0 4 1 4 9 5 9]
Actual:      [7 2 1 0 4 1 4 9 5 9]

Full test predictions logged in mnist_test_log.txt.

Requirements

Python 3.9+

NumPy

python-mnist

pip install numpy python-mnist

Usage

Clone the repository and download MNIST data:
git clone <repo-url>
cd <repo>
mkdir mnist_data
# Download MNIST data into ./mnist_data using the python-mnist helper scripts
1. Download the MNIST Dataset

You can download the MNIST dataset directly from the official source using wget:
```
cd ./mnist_data
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
```

After downloading, unzip the files:
gunzip *.gz

Ensure the following files are present in your mnist_data directory:
train-images-idx3-ubyte,
train-labels-idx1-ubyte,
t10k-images-idx3-ubyte,
t10k-labels-idx1-ubyte



Run the training script:

python3 mnist_net.py

After training, the script prints test accuracy and writes full predictions vs actual labels to mnist_test_log.txt.

Code Structure

mnist_net.py – main CNN training and evaluation script.

mnist_test_log.txt – generated after evaluation, contains predictions and actual labels.

Network Architecture

Conv layer: 8 filters, 3x3, ReLU activation

Max-pooling: 2x2

Fully connected: 128 hidden units

Output: 10 classes with softmax

Optimizers

Adam + SGD with momentum applied simultaneously.

Why This Project Matters

Shows the mechanics of a CNN from scratch, without ML libraries.

Ideal for learning forward/backward passes, convolution, pooling, and optimizer mechanics.

A foundation for more complex neural networks, including multi-layer CNNs or sequence models.

Potential Improvements

Increase network depth for higher accuracy.

Add dropout or batch normalization for better generalization.

Extend to other datasets (Fashion-MNIST, CIFAR-10).

Replace dual optimizer with single modern optimizer for simplicity.



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('train.csv/train.csv')
data = np.array(data) 
m, n = data.shape
np.random.shuffle(data)

def init_params():
    W1 = np.random.randn(10, 784) - 0.5;
    b1 = np.random.randn(10, 1) - 0.5;
    W2 = np.random.randn(10, 10) - 0.5;
    b2 = np.random.randn(10, 1) -0.5;
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))
    
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1;
    A1 = ReLU(Z1);
    Z2 = W2.dot(A1) + b2;
    A2 = softmax(Z2)
    return A2
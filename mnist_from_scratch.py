import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('train.csv/train.csv')
print(data.head()) 
data = np.array(data)
m, n = data.shape
print(m,n)
np.random.shuffle(data)

#Splitting the dataset
trainingSet = data[1000:m].T
trainY = trainingSet[0]
trainX = trainingSet[1:n] 
trainX = trainX / 255

validationSet = data[0:1000].T 
validationY = validationSet[0]
validationX = validationSet[1:n]
validationX = validationX / 255


def init_params():
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10) 
    b2 = np.random.randn(10, 1) 
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
    return Z1, A1, Z2, A2

# 1 => [0 1 0 0 0 0 0 0 0 0]
def labelToArray(Y):
    labelArray = np.zeros((Y.size, Y.max() + 1))
    labelArray[np.arrange(Y.size), Y] = 1

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        
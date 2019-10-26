# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 20:23:29 2019

@author: K1L4
"""


import numpy as np
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt

xtrain, ytrain = make_regression(n_samples=1000, n_features=1, n_informative=1, random_state=0, noise=10)

ytrain = np.array(ytrain).reshape(1000,1)
theta , bias = np.random.uniform(-1,1,(1,1)),np.random.uniform(-1,1)


def forward(x_train,theta,bias):
    
    predictions = [np.sum(np.multiply(theta,i))+bias for i in x_train]
    
    return np.array(predictions).reshape(x_train.shape[0],1)

def modification(predictions,y_train,x_train,theta,bias,lr):
    errors = np.subtract(predictions,y_train)
    deltheta = (2/len(errors))*np.sum(np.dot(np.transpose(x_train),errors))
    delbias = (2/len(errors))*np.sum(errors)
    theta = theta - (lr * deltheta)
    bias = bias - (lr * delbias)
    
    return theta, bias

def cost(prediction,ytrain):
    err = (1/len(ytrain))*np.sum(np.power(np.subtract(prediction,ytrain),2))
    return err
    
for i in range(50):
    result = forward(xtrain,theta,bias)
    theta,bias = modification(result,ytrain,xtrain,theta,bias,0.1)
    err = cost(result,ytrain)
    print(err)
    
plt.scatter(xtrain,ytrain)
plt.plot(xtrain, theta*xtrain+bias, 'r')
plt.show()
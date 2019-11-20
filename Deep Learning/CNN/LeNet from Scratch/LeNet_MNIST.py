# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:44:04 2019

@author: K1L4
"""
#Defining Relu function
def relu(a,derivative = False):
    if derivative == True:
        return 1*(a>0)
    return a*(a>0)
def sigmoid(b):
    return np.exp(b)/np.sum(np.exp(b))
import numpy as np
from numpy import flip,unravel_index
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
X_train = np.array(x_train)/255
X_train = np.concatenate((np.zeros((60000,2,32)),np.concatenate((np.zeros((60000,28,2)),X_train,np.zeros((60000,28,2))),axis = 2),np.zeros((60000,2,32))),axis = 1)
Y_train = np.array(y_train)
lb = preprocessing.LabelBinarizer()
Y_train = lb.fit_transform(Y_train)

X_test = np.array(x_test)/255
X_test = np.concatenate((np.zeros((10000,2,32)),np.concatenate((np.zeros((10000,28,2)),X_test,np.zeros((10000,28,2))),axis = 2),np.zeros((10000,2,32))),axis = 1)
Y_test = np.array(y_test).reshape(10000,1)
Y_t = lb.fit_transform(Y_test)
#Forward Propagation

#Layer1 Convolution layer
Ker1 = np.random.uniform(-1/np.sqrt(6*25),1/np.sqrt(6*25),(6,5,5))
#Ker1 = np.random.uniform(-1/np.sqrt(6*25),1/np.sqrt(6*25),(6,5,5))
Ker2 = np.random.uniform(-1/np.sqrt(6*25),1/np.sqrt(6*25),(16,6,5,5))
#Ker2 = np.random.uniform(-1/np.sqrt(6*25),1/np.sqrt(6*25),(16,6,5,5))
Ker3 = np.random.uniform(-1/np.sqrt(16*25),1/np.sqrt(16*25),(120,16,5,5)) 
#Ker3 = np.random.uniform(-1/np.sqrt(16*25),1/np.sqrt(16*25),(120,16,5,5)) 
#theta1 = np.random.uniform(-1,1,(84,120))
theta1 = np.random.uniform(-1/np.sqrt(120),1/np.sqrt(120),(84,120))
#theta2 = np.random.uniform(-1,1,(10,84))
theta2 = np.random.uniform(-1/np.sqrt(84),1/np.sqrt(84),(10,84))
out1 = np.zeros((6,28,28))
out2 = np.zeros((6,14,14))
out3 = np.zeros((16,10,10))
out4 = np.zeros((16,5,5))

out6 = np.zeros((84,1))
out = np.zeros((10,1))
for epo in range(2):
    for q in range(60000):
        inp = X_train[q]
        kernel1 = flip(flip(Ker1,-1),-2)
        
        for k in range(6):
            for i in range(28):
                for j in range(28):
                    out1[k,i,j] = np.sum(np.multiply(inp[i:5+i,j:5+j],kernel1[k]))
        
        #Layer2 Maxpooling Layer
        
        for k in range(6):
            for i in range(14):
                for j in range(14):
                    out2[k,i,j] = np.max(out1[k,2*i:2+2*i,2*j:2+2*j])
        out2 = relu(out2)
        
        #Layer3 Convolution Layer
        
        kernel2 = flip(flip(Ker2,-1),-2)
        
        for k in range(16):
            for i in range(10):
                for j in range(10):
                    out3[k,i,j] = np.sum(np.multiply(out2[0:6,i:5+i,j:5+j],kernel2[k]))
        
        #Layer4 MaxPooling Layer
        
        for k in range(16):
            for i in range(5):
                for j in range(5):
                    out4[k,i,j] = np.max(out3[k,2*i:2+2*i,2*j:2+2*j])
        out4 = relu(out4)
        
        #Layer5 Convolutional Layer
        
        kernel3 = flip(flip(Ker3,-1),-2)
        out5 = np.zeros((120,1,1))
        for k in range(120):
            out5[k,0,0] = np.sum(np.multiply(out4,kernel3[k]))
        #Flattening the layer
        out5 = relu(out5.reshape(120,1))
        
        
        #Layer6 Fully Connected Layer
        
        
        out6 = relu(np.dot(theta1,out5))
        
        #Layer7 Output Layer with 10 classes
        
        
        out = sigmoid(np.dot(theta2,out6))
        
        #Backward Propagation
        lr = 0.0001
        y = Y_train[q].reshape(10,1)
        e = out-y
        delout6 = np.multiply(np.dot(np.transpose(theta2),e),relu(out6,derivative=True))
        delout5 = np.multiply(np.dot(np.transpose(theta1),delout6),relu(out5,derivative=True)).reshape(120,1,1)
        delout4 = np.zeros((16,5,5))
        delout5pad = np.concatenate((np.zeros((120,4,9)),np.concatenate((np.zeros((120,1,4)),delout5,np.zeros((120,1,4))),axis = 2),np.zeros((120,4,9))),axis = 1)
        for k in range(16):
            for i in range(5):
                for j in range(5):
                    delout4[k,i,j] = np.sum(np.multiply(Ker3[0:120,k,0:5,0:5],delout5pad[0:120,i:5+i,j:5+j]))
        delout4 = np.multiply(delout4,relu(out4,derivative=True))
        delout3 = np.zeros((16,10,10))
        for k in range(16):
            for i in range(5):
                for j in range(5):
                    place = unravel_index(np.argmax(out3[k,2*i:2*i+2,2*j:2*j+2]),(2,2))
                    delout3[k,2*i+place[0],2*j+place[1]] = delout4[k,i,j]
        delout3pad = np.concatenate((np.zeros((16,4,18)),np.concatenate((np.zeros((16,10,4)),delout3,np.zeros((16,10,4))),axis = 2),np.zeros((16,4,18))),axis = 1)
        delout2 = np.zeros((6,14,14))
        for k in range(6):
            for i in range(14):
                for j in range(14):
                    delout2[k,i,j] = np.sum(np.multiply(Ker2[0:16,k,0:5,0:5],delout3pad[0:16,i:5+i,j:5+j]))
        delout1 = np.zeros((6,28,28))
        for k in range(6):
            for i in range(14):
                for j in range(14):
                    place = unravel_index(np.argmax(out1[k,2*i:2*i+2,2*j:2*j+2]),(2,2))
                    delout1[k,2*i+place[0],2*j+place[1]] = delout2[k,i,j]
        #Updating the weights
        inpf = flip(flip(inp,axis = -1),axis = -2)
        delKer1 = np.zeros((6,5,5))
        for k in range(6):
            for i in range(5):
                for j in range(5):
                    delKer1[k,i,j] = np.sum(np.multiply(delout1[k],inpf[i:28+i,j:28+j]))
        
        delKer2 = np.zeros((16,6,5,5))
        out2f = flip(flip(out2,axis = -1),axis = -2)
        for k in range(16):
            for l in range(6):
                for i in range(5):
                    for j in range(5):
                        delKer2[k,l,i,j] = np.sum(np.multiply(delout3[k],out2f[l,i:10+i,j:10+j]))
        delKer3 = np.zeros((120,16,5,5))
        out4f = flip(flip(out4,axis = -1),axis = -2)
        for k in range(120):
            for l in range(16):
                for i in range(5):
                    for j in range(5):
                        delKer3[k,l,i,j] = np.multiply(delout5[k],out4f[l,i:i+1,j:j+1])
        deltheta1 = np.dot(delout6,np.transpose(out5))
        deltheta2 = np.dot(e,np.transpose(out6))
        
        #Updating the weights
        Ker1 = (Ker1) - (lr*delKer1)
        Ker2 = (Ker2) - (lr*delKer2)
        Ker3 = (Ker3) - (lr*delKer3)
        theta1 = (theta1) - (lr*deltheta1)
        theta2 = (theta2) - (lr*deltheta2)  
        if(q%100) == 0:
            crossentropy = -np.log(out[np.argmax(Y_train[q])])
            print(crossentropy,epo,q)

f = open('LeNet_MNIST_trail.txt','wb')
pickle.dump([Ker1,Ker2,Ker3,theta1,theta2],f)
f.close()
resultf = []        
for q in range(10000):
    inp = X_test[q]
    kernel1 = flip(flip(Ker1,-1),-2)
    
    for k in range(6):
        for i in range(28):
            for j in range(28):
                out1[k,i,j] = np.sum(np.multiply(inp[i:5+i,j:5+j],kernel1[k]))
    
    #Layer2 Maxpooling Layer
    
    for k in range(6):
        for i in range(14):
            for j in range(14):
                out2[k,i,j] = np.max(out1[k,2*i:2+2*i,2*j:2+2*j])
    out2 = relu(out2)
    
    #Layer3 Convolution Layer
    
    kernel2 = flip(flip(Ker2,-1),-2)
    
    for k in range(16):
        for i in range(10):
            for j in range(10):
                out3[k,i,j] = np.sum(np.multiply(out2[0:6,i:5+i,j:5+j],kernel2[k]))
    
    #Layer4 MaxPooling Layer
    
    for k in range(16):
        for i in range(5):
            for j in range(5):
                out4[k,i,j] = np.max(out3[k,2*i:2+2*i,2*j:2+2*j])
    out4 = relu(out4)
    
    #Layer5 Convolutional Layer
    
    kernel3 = flip(flip(Ker3,-1),-2)
    out5 = np.zeros((120,1,1))
    for k in range(120):
        out5[k,0,0] = np.sum(np.multiply(out4,kernel3[k]))
    #Flattening the layer
    out5 = relu(out5.reshape(120,1))
    
    
    #Layer6 Fully Connected Layer
    
    
    out6 = relu(np.dot(theta1,out5))
    
    #Layer7 Output Layer with 10 classes
    
    
    out = sigmoid(np.dot(theta2,out6))
    resultf.append(np.argmax(out))   

resultf = np.array(resultf).reshape(10000,1)
mat = confusion_matrix(Y_test,resultf)
acc = accuracy_score(Y_test,resultf)
print(acc)
#converting confusion matrix into a dataframe
df_cm = pd.DataFrame(mat, index = [i for i in "0123456789"],columns = [i for i in "0123456789"])

#plotting the confusion matrix using seaborn library
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
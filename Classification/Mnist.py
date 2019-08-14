import torch # used to extract data from .pt files
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#Extract data from pt files
(x_train, y_train, x_test, y_test)=torch.load('mnist.pt')

#Convert X_train and Y_train into numpy array
X_train = np.array(x_train).reshape(1000,784)#reshaping into 1000 rows of 784 feature vector
Y_train = np.array(y_train).reshape(1000,1)#reshaping into 1000 rows

#Convert X_test and Y_test into numpy array
X_test = np.array(x_test).reshape(100,784)#reshaping X_test into rows of 784 feature vector 
Y_test = np.array(y_test).reshape(100,1)#reshaping Y_test into rows of 100 rows 

k = 5 # define k

temp = [] # array created to store euclidean distances of a test image to all train images, it refreshes after each test image
resultf = [] # array created to store the final predictions

for i in X_test:
    for j in X_train:
        temp.append(np.linalg.norm(i-j)) #appending euclidean dist to temp
    sort = np.argsort(temp)
    resulti = np.zeros(10) # creating a classification vector to count the nearest k neighbours
    for p in range(k):
        resulti[Y_train[sort[p]]]+=1 # counting the repitition of a class in k neighbours
    resultf.append(np.argmax(resulti)) # appending the latest result to the final array
    temp = []

# reshaping the final predictions into a 100*1 vector
resultf = np.array(resultf).reshape(100,1)
  
# creating confusion matrix using predictions and y_test
mat = confusion_matrix(Y_test,resultf)
# measuring accuracy of the predictions
acc = accuracy_score(Y_test,resultf)
print(acc)

#converting confusion matrix into a dataframe
df_cm = pd.DataFrame(mat, index = [i for i in "0123456789"],columns = [i for i in "0123456789"])

#plotting the confusion matrix using seaborn library
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

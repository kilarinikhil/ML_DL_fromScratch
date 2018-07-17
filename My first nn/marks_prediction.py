import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2,9],[1,5],[3,6]),dtype = float)
y = np.array([92],[86],[89],dtype = float)

# Scale units
X = X/np.amax(X, axis = 0) #columnwise max of X
y = y/100 # maximum marks are 100

class Neural_Network(object):
    def

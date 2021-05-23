#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:06:38 2021

@author: elham
"""

import numpy as np
from numpy import random

l = [1,2,3]

lenght = len(l)

print('hello')
print(type(l))
##numpy vectors 
## rank 1 array 
x = random.rand(5)
print(x)

z = random.randint(100,size=5,dtype=int)
print(z.shape)

print(z.T)
print(np.dot(z,z.T))

## column vector 
a = np.random.rand(5,1)
print(a)
print(a.T)
print(np.dot(a,a.T))
##row vector 
a = np.random.rand(1,5)

# a = np.random.randn(4, 3) 
# b = np.random.randn(3, 2)

#c = a*b
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
print(c.shape)


## sigmoid
x = np.array([1,2,3])
def sigmoid(x):
    "x an array and function retun computed sigmoid of x"
    s = 1/(1+np.exp(-x))
    return s

print("sigmoid of x is"+str(sigmoid(x)))

## derivative of sigmoid

def sigmoid_derivative(x):
    "derivative s ===>ds=s(1-s)"
    s = sigmoid(x)
    ds = s * (1-s)
    return ds

print("sigmod_derivative[x] ="+str(sigmoid_derivative(x)))

def image_to_vector(image):
    """image in shape(height,width,depth)===>
        reshape(height*width*depth,1)
    """
    v =  image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    
    return v

image = np.array([[[1,2,3]],[[1,3,4]],[[3,4,2]]])

print("image_to_vector"+str(image_to_vector(image)))


###  normalization in numpy

def normalize_row(x):
    """
    this function take x and normalized it by row
    """

    x_norm = np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    x = x/x_norm
    return x

x = np.array([[1,2,3],[2,3,1]])
print('shape x'+str(x.shape)+"and normalize x"+str(normalize_row(x)))

##Implement a softmax function using numpy
def softmax(x):
    """Calculates the softmax for each row of the input x.
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    exp_x = np.exp(x)
    sum_x = np.sum(x,axis=1,keepdims=True)
    s = exp_x/sum_x
    return s

t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(t_x)))

##Implement the numpy vectorized version of the L1 loss
def loss1(yhat,y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    abs_y = np.abs(yhat-y)
    loss = np.sum(abs_y)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(loss1(yhat, y)))



def loss2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function 
    """
    
    y_diff = np.abs(yhat-y)
    y_squre =np.dot(y_diff,y_diff)
    loss = np.sum(y_squre)
    
    
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(loss2(yhat, y)))

def propagate(w,b,X,Y):
    
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    Z= np.dot(w.T,X)+b
    print("Z", Z.shape)
    A = sigmoid(Z)
    print("A", A.shape)
    print("Y", Y.shape)
    
    cost = np.sum(np.dot(Y,np.log(A).T)+ np.dot((1-Y),np.log(1-A).T))/-m
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m  
    
    cost = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    return  grads, cost 

w =  np.array([[1.], [2.]])
b = 2.
X =np.array([[1., 2., -1.], [3., 4., -3.2]])
Y = np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
    
 
## play with array and know their size    
 
w = np.array([[0.1124579], [0.23106775]])
print(w.shape)
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
print(X.shape)
w = w.reshape(X.shape[0],1)
print(w)

## how to flatten an nd array and transpose it
a_3d_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(a_3d_array.shape)
flatten = a_3d_array.reshape(a_3d_array.shape[0],-1).T
print(flatten.shape)


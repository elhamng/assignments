#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:08:39 2021

@author: elham
"""

import numpy as np
import time
from math import exp


### example that shows us how vectorization increae the performance of computation 
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()


print(c)
print("vectorization", str(1000*(toc-tic)) +"ms")

c = 0
tic=time.time()
for i in range(1000000):
    c +=a[i]*b[i]
toc = time.time()
print(c)
print("for loop", str(1000*(toc-tic)) +"ms")


#### exponensial
n = 10
v = np.random.rand(10)
z = np.zeros((n,0))
tic=time.time()
for i  in range(n):
    u[i] = exp(v[i])
toc = time.time()    
    
print(u)
print("for loop", str(1000*(toc-tic)) +"ms")
u = 0
tic=time.time()
u = np.exp(v)
toc = time.time() 
print(u)    
print("for loop", str(1000*(toc-tic)) +"ms")
    





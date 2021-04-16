#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:10:08 2020

@author: lalithsrinivas
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def hessian(theta, x):
    hess = np.zeros((len(theta), len(theta)))
    temp = np.dot(np.array(theta).transpose(), np.array(x))
    h_theta = (1/(1+np.e**(-1*temp)))
    # if h_theta > 1-(10**-5):
    #     h_theta -= 10**-5
    # else:
    #     h_theta += 10**-5
    # print(h_theta)
    # print(h_theta)
    for i in range(len(theta)):
        for j in range(len(theta)):
            # print(i, j)
            hess[i][j] = (-1)*h_theta*(1-h_theta)*x[i]*x[j]
            # print(hess[i][j])
    if(np.linalg.det(hess) == 0):
        hess = hess + (10**(-5))*np.identity(3)
    return hess


x = open('../data/data/q3/logisticX.csv', "r").read().split('\n')[:-1]
y = open('../data/data/q3/logisticY.csv', "r").read().split('\n')[:-1]
y = [float(i) for i in y]
x = [i.split(',') for i in x]
x1 = np.array([float(i[0]) for i in x])
x2 = np.array([float(i[1]) for i in x])
x0 = np.array([1]*len(x1))

mean = 0
for i in x1:
    mean += float(i)
mean /= len(x1)

var = 0
for i in x1:
    var += (float(i) - mean)**2
var /= len(x1)

for i in range(len(x1)):
    x1[i] = (x1[i]-mean)/(var**0.5)

mean = 0
for i in x2:
    mean += float(i)
mean /= len(x2)

var = 0
for i in x2:
    var += (float(i) - mean)**2
var /= len(x2)

for i in range(len(x2)):
    x2[i] = (x2[i]-mean)/(var**0.5)


theta = np.array([0, 0, 0])
prev_theta = theta
l_prev = float('inf')
l = 0
iterations = 0
while(abs(l-l_prev) > 10**(-5)):
    sum_gradients = 0
    iterations += 1
    l_prev = l
    l = 0
    for i in range(len(y)):
        temp = (1/(1+(np.e**(-1*(theta[0]+theta[1]*x1[i]+theta[2]*x2[i])))))
        temp += 10**-5
        # print(temp)
        gradient = (y[i]-temp)*np.array([x0[i], x1[i], x2[i]])
        l += y[i]*(math.log(temp))+(1-y[i])*(math.log(abs(1-temp)))
        # print([x0[i], x1[i], x2[i]])
        hess_inv = np.linalg.inv(hessian(theta, [x0[i], x1[i], x2[i]]))
        theta = theta - np.dot(hess_inv, gradient)
    print(l)
    # print(theta, max(abs(max(prev_theta-theta)), abs(min(prev_theta-theta))))

print(theta)
h = [0]*len(y)
for i in range(len(y)):
    h[i] = (1/(1+(np.e**(-1*(theta[0]+theta[1]*x1[i]+theta[2]*x2[i])))))

x_10, x_20 = [], []
x_11, x_21 = [], []
for i in range(len(y)):
    if y[i]==0:
        x_10.append(x1[i])
        x_20.append(x2[i])
    else:
        x_11.append(x1[i])
        x_21.append(x2[i])


plt.scatter(x_10, x_20)
plt.scatter(x_11, x_21)
plt.plot(x1, (-1/(theta[2]/1000))*((theta[1]/1000)*x1+(theta[0]/1000)))
plt.show()
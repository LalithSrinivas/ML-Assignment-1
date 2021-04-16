#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:06:21 2020

@author: lalithsrinivas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.random.normal(3, 2, 1000000)
x2 = np.random.normal(-1, 2, 1000000)
epsilon = np.random.normal(0, 2**0.5, 1000000)
theta = [3, 1, 2]
y = theta[0]+epsilon+theta[1]*x1+theta[2]*x2
theta = [0, 0, 0]
b_size = 1
batch_error = float('inf')
prev_error = float('inf')
alpha = 0.001
iteration = 0
prev_batch_error = 0
error = 0
thetaList1 = []
thetaList2 = []
thetaList3 = []
thetaList4 = []

while(abs(error-prev_error) > 10**(-5)):
    iteration += 1
    print(iteration)
    for i in range(0, 1000000, b_size):
        prev_error = error
        if i!= 0:
            error = (b_size*batch_error+(i-b_size)*prev_error)/i
        cond = float('inf')
        # while(abs(cond) > 10**(-2)):
        prev_batch_error = batch_error
        sum1 = 0
        sum2 = 0
        sum3 = 0
        batch_error = 0
        for k in range(i, i+b_size):
            sum1 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))
            sum2 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x1[k]
            sum3 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x2[k]
            batch_error = batch_error + ((y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))**2)
        batch_error = batch_error/(2*b_size)
        theta[0] = theta[0] + alpha*sum1
        theta[1] = theta[1] + alpha*sum2
        theta[2] = theta[2] + alpha*sum3
        cond = (prev_batch_error - batch_error)
        thetaList1.append(theta)
        # print(cond, i)

print("num of iterations: ", iteration)
iteration = 0
print("batch size: 1, theta: ", theta)

theta = [0, 0, 0]
b_size = 100
batch_error = float('inf')
prev_error = float('inf')
alpha = 0.001
iteration = 0
prev_batch_error = 0
error = 0
while(abs(error-prev_error) > 10**(-2)):
    iteration += 1
    for i in range(0, 1000000, b_size):
        # print(iteration)
        prev_error = error
        if i!= 0:
            error = (b_size*batch_error+(i-b_size)*prev_error)/i
        cond = float('inf')
        # while(abs(cond) > 10**(-2)):
        prev_batch_error = batch_error
        sum1 = 0
        sum2 = 0
        sum3 = 0
        batch_error = 0
        for k in range(i, i+b_size):
            sum1 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))
            sum2 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x1[k]
            sum3 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x2[k]
            batch_error = batch_error + ((y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))**2)
        batch_error = batch_error/(2*b_size)
        theta[0] = theta[0] + alpha*sum1
        theta[1] = theta[1] + alpha*sum2
        theta[2] = theta[2] + alpha*sum3
        cond = (prev_batch_error - batch_error)
        thetaList2.append(theta)
            # print(cond, i)

print("num of iterations: ", iteration)
iteration = 0
print("batch size: 100, theta: ", theta)

theta = [0, 0, 0]
b_size = 10000
batch_error = float('inf')
prev_error = float('inf')
alpha = 0.00001
iteration = 0
prev_batch_error = 0
error = 0
while(abs(error-prev_error) > 10**(-2)):
    iteration += 1
    for i in range(0, 1000000, b_size):
        # print(iteration)
        prev_error = error
        if i!= 0:
            error = (b_size*batch_error+(i-b_size)*prev_error)/i
        cond = float('inf')
        # while(abs(cond) > 10**(-2)):
        prev_batch_error = batch_error
        sum1 = 0
        sum2 = 0
        sum3 = 0
        batch_error = 0
        for k in range(i, i+b_size):
            sum1 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))
            sum2 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x1[k]
            sum3 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x2[k]
            batch_error = batch_error + ((y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))**2)
        batch_error = batch_error/(2*b_size)
        theta[0] = theta[0] + alpha*sum1
        theta[1] = theta[1] + alpha*sum2
        theta[2] = theta[2] + alpha*sum3
        cond = (prev_batch_error - batch_error)
            # print(cond, i, sum1, sum2, sum3)
        thetaList3.append(theta)


print(abs(error-prev_error), error, prev_error)
print("num of iterations: ", iteration)
iteration = 0
print("batch size: 10000, theta: ", theta)

theta = [0, 0, 0]
b_size = 1000000
batch_error = float('inf')
prev_error = float('inf')
alpha = 0.0000001
iteration = 0
prev_batch_error = 0
error = 0
while(abs(error-prev_error) > 10**(-2)):
    iteration += 1
    for i in range(0, 1000000, b_size):
        # print(iteration)
        prev_error = error
        error = batch_error
        cond = float('inf')
        # while(abs(cond) > 10**(-2)):
        prev_batch_error = batch_error
        sum1 = 0
        sum2 = 0
        sum3 = 0
        batch_error = 0
        for k in range(i, i+b_size):
            sum1 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))
            sum2 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x1[k]
            sum3 += (y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))*x2[k]
            batch_error = batch_error + ((y[k]-(theta[0]+theta[1]*x1[k]+theta[2]*x2[k]))**2)
        batch_error = batch_error/(2*b_size)
        theta[0] = theta[0] + alpha*sum1
        theta[1] = theta[1] + alpha*sum2
        theta[2] = theta[2] + alpha*sum3
        cond = (prev_batch_error - batch_error)
            # print(cond, i, sum1, sum2, sum3)
        thetaList4.append(theta)
print(abs(error-prev_error), error, prev_error)              
print("num of iterations: ", iteration)
iteration = 0
print("batch size: 1000000, theta: ", theta)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_ylabel('theta_1')
ax.set_xlabel('theta_0')
ax.set_zlabel('theta_2')

ax.plot(np.array(thetaList4)[:, 0], np.array(thetaList4)[:, 1], np.array(thetaList4)[:, 2])
plt.draw()
plt.show()


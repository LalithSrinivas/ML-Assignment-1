#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:22:36 2020

@author: lalithsrinivas
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
x = open('../data/data/q1/linearX.csv', "r").read().split('\n')[:-1]
y_file = open('../data/data/q1/linearY.csv', "r").read().split('\n')[:-1]
y_file = [float(i) for i in y_file]
x = [float(i) for i in x]
theta = [0, 0]
alpha = 0.005
error = float('inf')
prev_err = 0
num_it = 0
m = len(x)
mean = 0
for i in x:
    mean += float(i)
mean /= m

var = 0
for i in x:
    var += (float(i) - mean)**2
var /= m

for i in range(m):
    x[i] = (x[i]-mean)/(var**0.5)
error_list = []
theta_0 = []
theta_1 = []
while(abs(error-prev_err) > 10**(-10)):
    num_it += 1
    prev_err = error
    sum1 = 0
    sum2 = 0
    error = 0
    for i in range(m):
        sum1 += (y_file[i]-(theta[0]+theta[1]*x[i]))
        sum2 += (y_file[i]-(theta[0]+theta[1]*x[i]))*x[i]
        error = error + ((y_file[i]-(theta[0]+theta[1]*x[i]))**2)
    error = error/(2*m)
    theta[0] = theta[0]+alpha*sum1
    theta[1] = theta[1] + alpha*sum2
    # if error < 0.1:
    error_list.append(error)
    theta_0.append(theta[0])
    theta_1.append(theta[1])

def err(a, b):
    error = 0
    for i in range(m):
        error = error + ((y_file[i]-(a+b*x[i]))**2)
    return error
print("theta: ", theta, " \nerror: ", error)
print("learning rate: ", alpha)
print("number of iterations: ", num_it)
z = [[0]*len(error_list)]*len(error_list)
for k in range(len(error_list)):
    for j in range(len(error_list)):
        sum1 = 0
        sum2 = 0
        error = 0
        for i in range(m):
            error = error + ((y_file[i]-(theta_0[k]+theta_1[j]*x[i]))**2)
        z[k][j] = error/(2*m)

X = np.linspace(0, 2, len(theta_0))
Y = np.linspace(-1, 1, len(theta_0))
X, Y = np.meshgrid(X, Y)
j = err(X, Y)
for i in range(len(theta_0)):
    # print()
    plt.contour(X, Y, j, levels=[err(theta_0[i], theta_1[i])])
    # plt.plot(np.array([theta_0[:i+1]]), np.array([theta_1[:i+1]]))
    # plt.ylim(1, 1.5)
    # plt.xlim(0, 0.5)
    # plt.autoscale(True)
    plt.draw()
#     plt.show()
    plt.pause(1e-10)
    time.sleep(0.2)
plt.show()

plt.scatter(x, y_file, color='r')
plt.plot(x, theta[0]*np.array([1]*m)+theta[1]*np.array(x), color = 'b')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_ylabel('theta_1')
ax.set_xlabel('theta_0')
ax.set_zlabel('lms')
ax.plot_wireframe(X, Y, j)
for i in range(len(error_list)):
    er = err(np.array(theta_0[:i+1]), np.array(theta_1[:i+1]))
    ax.plot(theta_0[:i+1], theta_1[:i+1], er, color='r')
    # ax.set_zlim(min(error_list), max(error_list))    
    # ax.set_ylim(min(theta_1), max(theta_1))
    # ax.set_xlim(min(theta_0), max(theta_0))
    # ax.autoscale(False)
    plt.draw()
#     plt.show()
    plt.pause(1e-10)
    time.sleep(0.2)
    # print(theta_0[:i+1], theta_1[:i+1], error_list[:i+1])
plt.show()
print("done")
plt.show()


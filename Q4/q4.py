#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:15:39 2020

@author: lalithsrinivas
"""

import numpy as np
import matplotlib.pyplot as plt

x = open('../data/data/q4/q4x.dat', "r").read().split('\n')[:-1]
y = open('../data/data/q4/q4y.dat', "r").read().split('\n')[:-1]
x = [i.split('  ') for i in x]
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


count=0
mu_0 = np.array([0, 0])
mu_1 = np.array([0, 0])
x1_alaska = []
x2_alaska = []
x1_Ntalaska = []
x2_Ntalaska = []
for i in range(len(y)):
    if y[i] == "Alaska":
        count+=1
        mu_1 = mu_1+np.array([x1[i], x2[i]])
        x1_alaska.append(x1[i])
        x2_alaska.append(x2[i])
    else:
        mu_0 = mu_0 + np.array([x1[i], x2[i]])
        x1_Ntalaska.append(x1[i])
        x2_Ntalaska.append(x2[i])

phi = count/len(y)
mu_0 = (1/(len(y)-count))*mu_0
mu_1 = (1/(count))*mu_1

sigma = np.array([[0, 0], [0, 0]])

for i in range(len(y)):
    mu = mu_0
    if y[i] == 'Alaska':
        mu = mu_1
    temp = np.array([x1[i], x2[i]])-mu
    temp = np.mat(temp)
    sigma = sigma+temp.T*temp

sigma = sigma*(1/len(y))

sigma0 = np.array([[0, 0], [0, 0]])
sigma1 = np.array([[0, 0], [0, 0]])
for i in range(len(y)):
    if y[i] != 'Alaska':
        mu = mu_1
        temp = np.array([x1[i], x2[i]])-mu
        temp = np.mat(temp)
        sigma0 = sigma0+temp.T*temp
        # print(temp.T*temp)
    else:
        mu = mu_0
        temp = np.array([x1[i], x2[i]])-mu
        temp = np.mat(temp)
        sigma1 = sigma1+temp.T*temp


sigma0 = (1/(len(y)-count))*sigma0
sigma1 = (1/count)*sigma1

print("sigma ", sigma)
print("sigma0 ", sigma0)
print("sigma1 ", sigma1)
print("mu0: ", mu_0)
print("mu1: ", mu_1)
print("phi: ", phi)

def gen_gda(x, y, mu0, mu1, sigma0, sigma1):
    inv_cov0 = np.linalg.pinv(sigma0)
    inv_cov1 = np.linalg.pinv(sigma1)
    theta = np.zeros(6)
    theta[0] = np.log(phi/(1-phi))
    temp1 = np.dot(inv_cov1, mu1)
    temp2 = np.dot(inv_cov0, mu0)
    for i in range(2):
        theta[0] -= (mu1[i]*temp1[0, i])/2
        theta[0] += (mu0[i]*temp2[0, i])/2
    theta[1] = temp1[0, 0]-temp2[0, 0]
    theta[2] = temp1[0, 1]-temp2[0, 1]
    theta[3] = -0.5*(inv_cov1[0, 0] - inv_cov0[0, 0])
    theta[4] = -0.5*(inv_cov1[1, 1] - inv_cov0[1, 1])
    theta[5] = -0.5*(inv_cov1[0, 1]+inv_cov1[1, 0]-inv_cov0[0, 1]-inv_cov0[1, 0])
    print(theta)
    return theta
linear = gen_gda(x, y, mu_0, mu_1, sigma, sigma)
quad = gen_gda(x, y, mu_0, mu_1, sigma0, sigma1)
plt.scatter(x1_alaska, x2_alaska, label='Alaska')
plt.scatter(x1_Ntalaska, x2_Ntalaska, label='Canada')
plt.legend()
plt.show()

plt.scatter(x1_alaska, x2_alaska, label='Alaska')
plt.scatter(x1_Ntalaska, x2_Ntalaska, label='Canada')
plt.plot(x1, (-1/linear[2])*(linear[0]+linear[1]*x1), label='Decision Boundary')
plt.legend(loc=4)
plt.show()

a = np.linspace(-3, 3, 60)
b = np.linspace(-3, 3, 60)
A, B = np.meshgrid(a, b)
F = quad[3]*A**2 + quad[5]*A*B + quad[4]*B**2 + quad[2]*B + quad[1]*A + quad[0]
plt.contour(A, B, F, levels=[0], label='Decision Boundary')
plt.scatter(x1_alaska, x2_alaska, label='Alaska')
plt.scatter(x1_Ntalaska, x2_Ntalaska, label='Canada')
plt.legend(loc=4)
plt.show()
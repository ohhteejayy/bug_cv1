#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:29:17 2020

@author: ros
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

#read image
img = cv2.imread('sample_test_negative.jpg')
img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)

#convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#get rgb values from image to 1D array
h, s, v = cv2.split(img)
h = h.flatten()
s = s.flatten()
v = v.flatten()

#plotting 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(h, s, v)
plt.show()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:57:38 2020

@author: ros
"""

import numpy as np
import cv2
import time

def clustering(sample_img):
    Z = sample_img.reshape((-1,3))
    Z = np.float32(Z)
    
    NUMBER_OF_CLUSTERS = 5
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                NUMBER_OF_CLUSTERS, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    t0 = time.time()
    compactness,labels,centers = cv2.kmeans(Z,NUMBER_OF_CLUSTERS,\
                                            None,criteria,1,flags)
    
    
    centers = np.uint8(centers)
    #res = centers[labels.flatten()]
    #res2 = res.reshape((sample_img.shape))
    
    '''
    We perform the MASKING operations on the Z to filter out pixels belong to 
    their corresponding labels
    Point estimates of distributions are mean and variance [[mean, variance]n]
    '''
    point_estimates = np.empty((0,2))
    for i in range(NUMBER_OF_CLUSTERS):
        label_index = (labels==i)
        boolean_index = np.dstack([label_index[:,0]]*3)[0]
        values_of_labels = Z[boolean_index].reshape((-1,3))
        var = np.var(values_of_labels)
        point_estimates = np.vstack((point_estimates, \
                                     np.array([[centers[i]], [var]]).T))
    #print(1/(time.time() - t0))
    
    return point_estimates

sample_img_positive = cv2.imread('sample_test_positive.jpg')
sample_img_postivie = cv2.cvtColor(sample_img_positive, cv2.COLOR_BGR2RGB)
sample_img_positive = cv2.resize(sample_img_positive, (0,0), fx=0.2, fy=0.2)
positive_point_estimates = clustering(sample_img_positive)

sample_img_negative = cv2.imread('sample_test_negative.jpg')
sample_img_negative = cv2.cvtColor(sample_img_negative, cv2.COLOR_BGR2RGB)
sample_img_negative = cv2.resize(sample_img_negative, (0,0), fx=0.2, fy=0.2)
negative_point_estimates = clustering(sample_img_negative)

from sklearn import svm

Y = np.full((5,1),1)
Y = np.concatenate((Y, np.full((5,1),0)))
Y = Y.tolist()
#Y = np.array([1,1,1,1,1,0,0,0,0,0]).tolist()
X = np.concatenate((positive_point_estimates[:,0], negative_point_estimates[:,0]))
ABC = []
for i in X:
    ABC.append(i.tolist())
clf = svm.SVC(kernel='rbf')
clf.fit(ABC,Y)

predicted_img = cv2.imread('test.jpg')
predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)
predicted_img = cv2.resize(predicted_img, (0,0), fx=0.2, fy=0.2)
Z = predicted_img.reshape((-1,3)).astype(np.uint8)
result = clf.predict(Z)

result = result.reshape((predicted_img.shape[0], predicted_img.shape[1]))
result = result*255
cv2.imwrite('test_result_RGB.jpg', result)

'''
from scipy.stats import multivariate_normal

input_fit = sample_img_positive.reshape((-1,3))
dis = multivariate_normal(mean=positive_point_estimates[1][0], cov=positive_point_estimates[1][1])
result = []
for i in input_fit:
    result.append(dis.pdf(i))

result = np.array(result)
print(np.max(result))
print(np.min(result))
'''


    
#sample_RGB = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)


#cv2.imshow('test', res2)
#cv2.waitKey(0)
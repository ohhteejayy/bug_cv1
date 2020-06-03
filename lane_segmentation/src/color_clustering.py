#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:57:38 2020

@author: ros
"""

import numpy as np
import cv2
import time
from scipy.stats import gamma
from scipy.stats import t


def clustering(sample_img):
    Z = sample_img.reshape((-1,3))
    Z = np.float32(Z)
    
    NUMBER_OF_CLUSTERS = 4
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                NUMBER_OF_CLUSTERS, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    global labels
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
    '''
    point_estimates = []
    for i in range(NUMBER_OF_CLUSTERS):
        label_index = (labels==i)
        boolean_index = np.dstack([label_index[:,0]]*3)[0]
        values_of_labels = Z[boolean_index].reshape((-1,3))
        
        #Covariance of three image channels calculations
        var0  = np.var(values_of_labels[:,0])
        var1  = np.var(values_of_labels[:,1])
        var2  = np.var(values_of_labels[:,2])

        point_estimates.append([centers[i], [var0,var1,var2]])
    #print(1/(time.time() - t0))
    '''
    
    point_estimates = []
    for i in range(NUMBER_OF_CLUSTERS):
        update_append = False
        similar_dist = None
        label_index = (labels==i)
        boolean_index = np.dstack([label_index[:,0]]*3)[0]
        values_of_labels = Z[boolean_index].reshape((-1,3))
        
        #Covariance of three image channels calculations
        var0  = np.var(values_of_labels[:,0])
        var1  = np.var(values_of_labels[:,1])
        var2  = np.var(values_of_labels[:,2])
        var   = np.array([var0, var1, var2])
        
        for j in range(len(point_estimates)):
            lower_50p = point_estimates[j]['50percentile'][:,0]
            upper_50p = point_estimates[j]['50percentile'][:,1]
            
            if not (np.count_nonzero(centers > lower_50p) & (np.count_nonzero(centers < upper_50p))):
                update_append = True
            else:
                similar_dist = j
        
        if (update_append==True and similar_dist==None) or (i==0):
            point_estimate = {}
            point_estimate['center'] = centers[i]
            point_estimate['variance'] = var
            point_estimate['50percentile'] = np.array([
                                           t.interval(0.50, 100, loc=centers[i][0], scale=var0),
                                           t.interval(0.50, 100, loc=centers[i][1], scale=var1),
                                           t.interval(0.50, 100, loc=centers[i][2], scale=var2)])   
            point_estimate['number_of_elements'] = 1             
            point_estimates.append(point_estimate)
            
        elif (update_append==False and similar_dist!=None):
            point_estimate = {}
            point_estimate['center'] = centers[i] + (centers[similar_dist]*i)/(i+1)
            point_estimate['variance'] = 
            point_estimate['number_of_elements'] = point_estimates[similar_dist]['number_of_elements'] + 1
            
    return point_estimates

''' 
Todo
Use color cluster to figure out most common color in a region from positive and negative samples
With their variance, pick the color range that minimize covariance
'''

predicted_img = cv2.imread('test.jpg')
hsv_predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2HSV)
hsv_threshold_sample = cv2.imread('sample_test_positive_1.jpg')
hsv_threshold_sample = cv2.cvtColor(hsv_threshold_sample, cv2.COLOR_BGR2HSV)
hsv_threshold_sample = hsv_threshold_sample.reshape((-1,3)).astype(np.uint8)
max_h = np.max(hsv_threshold_sample[:,0])
min_h = np.min(hsv_threshold_sample[:,0])
max_s = np.max(hsv_threshold_sample[:,1])
min_s = np.min(hsv_threshold_sample[:,1])
max_v = np.max(hsv_threshold_sample[:,2])
min_v = np.min(hsv_threshold_sample[:,2])

hsv_min = np.array([min_h, min_s, min_v], np.uint8)
hsv_max = np.array([max_h, max_s, max_v], np.uint8)

frame_thresh = cv2.inRange(hsv_predicted_img, hsv_min, hsv_max)
cv2.imwrite('output_pre_processed_2.jpg', frame_thresh)

sample_img_positive = cv2.imread('sample_test_positive_1.jpg')
sample_img_positive = cv2.cvtColor(sample_img_positive, cv2.COLOR_BGR2HSV)
sample_img_positive = cv2.resize(sample_img_positive, (0,0), fx=0.2, fy=0.2)
positive_point_estimates = clustering(sample_img_positive)

'''
sample_img_positive = cv2.imread('sample_test_positive_1.jpg')
sample_img_positive = cv2.cvtColor(sample_img_positive, cv2.COLOR_BGR2HSV)
sample_img_positive = cv2.resize(sample_img_positive, (0,0), fx=0.2, fy=0.2)
positive_point_estimates = np.concatenate((positive_point_estimates, \
                                     clustering(sample_img_positive)))
'''

sample_img_negative = cv2.imread('sample_test_negative.jpg')
sample_img_negative = cv2.cvtColor(sample_img_negative, cv2.COLOR_BGR2HSV)
sample_img_negative = cv2.resize(sample_img_negative, (0,0), fx=0.2, fy=0.2)
negative_point_estimates = clustering(sample_img_negative)

from sklearn import svm

Y = np.full((10,1),1)
Y = np.concatenate((Y, np.full((10,1),0)))
#Y = np.array([1,1,1,1,1,0,0,0,0,0]).tolist()
X = np.concatenate((positive_point_estimates[:,0], negative_point_estimates[:,0]))
ABC = []
XYZ = []
for i in X:
    ABC.append(i.tolist())
for i in Y:
    XYZ.append(i[0])
clf = svm.SVC(kernel='rbf')
clf.fit(ABC,XYZ)

predicted_img = cv2.imread('test.jpg')
predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2HSV)
predicted_img = cv2.resize(predicted_img, (0,0), fx=0.2, fy=0.2)
Z = predicted_img.reshape((-1,3)).astype(np.uint8)  
result = clf.predict(Z)

'''
result = result.reshape((predicted_img.shape[0], predicted_img.shape[1]))
result = result*255
cv2.imwrite('test_result_HSV.jpg', result)
'''

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
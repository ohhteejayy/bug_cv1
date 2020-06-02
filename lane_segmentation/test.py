#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:23:21 2020

@author: ros
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:57:38 2020

@author: ros
"""

import numpy as np
import cv2
import time
from scipy.stats import t


def clustering(sample_img):
    Z = sample_img.reshape((-1,3))
    Z = np.float32(Z)
    
    NUMBER_OF_CLUSTERS = 3
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
                NUMBER_OF_CLUSTERS, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    global labels
    global centers
    t0 = time.time()
    compactness,labels,centers = cv2.kmeans(Z,NUMBER_OF_CLUSTERS,\
                                            None,criteria,3,flags)
    
    
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
        label_index = (labels==i)
        boolean_index = np.dstack([label_index[:,0]]*3)[0]
        values_of_labels = Z[boolean_index].reshape((-1,3))
        
        #Covariance of three image channels calculations
        var0  = np.var(values_of_labels[:,0])
        var1  = np.var(values_of_labels[:,1])
        var2  = np.var(values_of_labels[:,2])
        var   = np.array(np.array([var0, var1, var2]))
        std   = np.sqrt(np.array([var0, var1, var2]))
    
        point_estimate = {}
        point_estimate['center'] = centers[i]
        point_estimate['standard_deviation'] = std
        point_estimate['90percentile'] = np.array([
                                       t.interval(0.90, 100, loc=centers[i][0], scale=var[0]),
                                       t.interval(0.90, 100, loc=centers[i][1], scale=var[1]),
                                       t.interval(0.90, 100, loc=centers[i][2], scale=var[2])])   
        point_estimates.append(point_estimate)
        
    return point_estimates

''' 
Todo
Use color cluster to figure out most common color in a region from positive and negative samples
With their variance, pick the color range that minimize covariance
'''

sample_img_positive = cv2.imread('sample_test_negative.jpg')
sample_img_positive = cv2.cvtColor(sample_img_positive, cv2.COLOR_BGR2HSV)
sample_img_positive = cv2.resize(sample_img_positive, (0,0), fx=0.2, fy=0.2)
positive_point_estimates = clustering(sample_img_positive)

predicted_img = cv2.imread('test.jpg')
hsv_predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2HSV)
output = np.zeros((predicted_img.shape[0], predicted_img.shape[1]))

for i in range(3):
    percentile_85 = positive_point_estimates[i]['90percentile']
    frame_thresh = cv2.inRange(hsv_predicted_img, percentile_85[:,0], percentile_85[:,1])
    output = output + frame_thresh
    
#output = cv2.GaussianBlur(output,(15,15),cv2.BORDER_DEFAULT)
cv2.imwrite('output_pre_processed.jpg', output)

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
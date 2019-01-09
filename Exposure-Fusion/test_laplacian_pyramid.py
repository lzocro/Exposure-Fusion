#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:26:41 2019

@author: bastien
"""

import cv2
import measures
import EF_utils
#import naive
import laplacian_pyramid as Lp
import numpy as np


img = cv2.imread('img_forest.jpg')
norm_img= img/255.
test = img[:,:,0]
norm_test =test/255.
depth = 8

# simple tests
l = []
for channel in range(3):
    l.append(np.uint8(np.round(Lp.Expand(norm_img[:,:,channel],0,0)*255.)))
d_image_r = np.stack(l, axis=-1)

l = []  
for channel in range(3):
    l.append(np.uint8(np.round(Lp.Expand(norm_img[:,:,channel],0,0)*255.)))
d_image_e = np.stack(l, axis=-1)


out = np.uint8(np.round(Lp.Expand(norm_test)*255.))
cv2.imshow('test', out)
cv2.imshow('test_color', d_image_e)

# Gaussian pyramid
g = Lp.Gaussian_pyramid(norm_test, depth)
i=0
for im in g:
    cv2.imshow('g_{}'.format(i), np.uint8(np.round(im*255.)))
    i+=1
  
# Laplacian pyramid
la = Lp.Laplacian_pyramid_raw(norm_test, depth)
for im in la:
    cv2.imshow('la_{}'.format(i), np.uint8(np.round(im*255.)))
    i+=1


# Rebuild image
u = Lp.Collapse_Laplacian_raw(la)
cv2.imshow('rebuild', np.uint8(np.round(u*255.)))

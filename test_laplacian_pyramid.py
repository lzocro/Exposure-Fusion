#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:26:41 2019

@author: bastien
"""

import cv2
import measures as me
import EF_utils as EF
#import naive
import laplacian_pyramid as Lp
import numpy as np


img = cv2.imread('venice_canal_exp_0.jpg')
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
  
# Laplacian pyramid grey scale
la = Lp.Laplacian_pyramid_raw(norm_test, depth)
for im in la:
    cv2.imshow('la_{}'.format(i), np.uint8(np.round(im*255.)))
    i+=1


# Rebuild image
u = Lp.Collapse_Laplacian_raw(la)
cv2.imshow('rebuild', np.uint8(np.round(u*255.)))



# several images
locations = ['venice_canal_exp_0.jpg', 'venice_canal_exp_1.jpg', 'venice_canal_exp_2.jpg']
w_exponents = [1,1,1]

image = EF.read_sequence_to_fuse(locations)
nb_image = len(locations)

# computes weights
W_norm = me.weight_calc(image, w_exponents)

# compute Gaussian pyramid of weights for each image
list_Gaus_pyr_w = []
for w in W_norm :
    g, depth = Lp.Gaussian_pyramid(w)
    list_Gaus_pyr_w.append(list(reversed((g))))
list_Gaus_pyr_w = [np.stack(i,axis=0) for i in zip(*list_Gaus_pyr_w)]

# compute Laplacian pyramid of each image
norm_img = image
list_Lapl_pyr = []
for im in norm_img :
    list_Lapl_pyr.append(Lp.Laplacian_pyramid(im, depth))
lapl_pyr = [np.stack(i,axis=0) for i in zip(*list_Lapl_pyr)]


l_lapl_fuse = []
for l in range(depth):
    R = np.zeros(lapl_pyr[l][0].shape, dtype=np.uint8)
    for i in range(nb_image):
        Gauss_stack = np.stack(3*[list_Gaus_pyr_w[l][i]], axis=-1)
        m = cv2.multiply(Gauss_stack, lapl_pyr[l][i], dtype=cv2.CV_8UC3)
        R = cv2.add(R,m)
    l_lapl_fuse.append(R)
    
hdr_im_norm =  Lp.Collapse_Laplacian(l_lapl_fuse)


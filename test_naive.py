#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:56:32 2019

@author: bastien
"""
import EF_utils as EF
import measures as m
from laplacian_pyramid import Gaussian_pyramid, Laplacian_pyramid, Collapse_Laplacian
import numpy as np
import cv2


locations = ['venice_canal_exp_0.jpg', 'venice_canal_exp_1.jpg', 'venice_canal_exp_2.jpg']
w_exponents = [1,1,1]
depth = 4
Max_range=255.

image = EF.read_sequence_to_fuse(locations)
W_norm = m.weight_calc(image, w_exponents)

list_Gaus_pyr_w = []
for w in W_norm :
    print(w.shape)
    l, max_depth = Gaussian_pyramid(w)
    list_Gaus_pyr_w.append(list(reversed((l))))
list_Gaus_pyr_w = [np.stack(i,axis=0) for i in zip(*list_Gaus_pyr_w)]
norm_img = image/Max_range
list_Lapl_pyr = []
for im in norm_img :
    list_Lapl_pyr.append(Laplacian_pyramid(im))

lapl_pyr = [np.stack(i,axis=0) for i in zip(*list_Lapl_pyr)]

l_lapl_fuse = []
for l in range(max_depth):
    l_lapl_fuse.append(np.einsum('lij,lijc->ijc',list_Gaus_pyr_w[l],lapl_pyr[l]))
    
hdr_im_norm =  Collapse_Laplacian(l_lapl_fuse)
hdr_im = np.uint8(np.round(hdr_im_norm*255.))

cv2.namedWindow('test',cv2.WINDOW_NORMAL)
image = cv2.resize(hdr_im, (600, 600))
cv2.imshow('test',hdr_im)
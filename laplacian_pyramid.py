#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:29:08 2018

@author: bastien
"""

import numpy as np
import cv2

# Global variables (Burt & Al kernel)
k = np.array([[.05, .25, .4, .25, .05]])
K = k.T.dot(k)


def Reduce(image):
    """
    Downsampling step for the Gaussian pyramid calculation
    Image values are btw [0,1]
    """
    # convolution step with kernel
    convol_image = cv2.filter2D(image,-1,K, borderType=cv2.BORDER_REFLECT_101) # half-symmetric boundary extension
    
    # resizing step (keeping even columns)
    new_image = convol_image[::2,::2]
    
    return new_image


def Expand(image,pad_h=0, pad_w=0):
    """
    Upsampling step for the Gaussian pyramid calculation
    Image values are btw [0,1]
    """
    im_border = cv2.copyMakeBorder(image,1,1,1,1, borderType=cv2.BORDER_REPLICATE)
    new_image = np.zeros((2*im_border.shape[0],2*im_border.shape[1]))
    new_image[::2,::2] = 4*im_border
    convol_image = cv2.filter2D(new_image,-1,K,borderType=cv2.BORDER_REFLECT_101)
    # dealing with odd values for image sizes
    convol_image = convol_image[2:,2:]
    convol_image = convol_image[:-(2+pad_h),:-(2+pad_w)]
    return convol_image


def Gaussian_pyramid(image, depth):
    """
    Construct the Gaussian pyramid with the desired depth
    Image values are btw [0,1]
    """
    gaus_pyr = []
    for d in range(depth):
        if d==0:
            gaus_pyr.append(image)
        else:
            p_image = gaus_pyr[-1]
            gaus_pyr.append(Reduce(p_image))
    return gaus_pyr


def Laplacian_pyramid_raw(image, depth):
    """
    Construct the Laplacian pyramid with the desired depth for a grey level image
    Image values are btw [0,1]
    """
    lapl_pyr = []
    gaus_pyr = Gaussian_pyramid(image, depth)
    for i in range(depth-1,-1,-1):
        if i==depth-1:
            lapl_pyr.append(gaus_pyr[i])
        else:
            pad_h = gaus_pyr[i].shape[0] % 2
            pad_w = gaus_pyr[i].shape[1] % 2
            l = gaus_pyr[i] - Expand(gaus_pyr[i+1], pad_h, pad_w)
            lapl_pyr.append(l)
    return lapl_pyr


def Laplacian_pyramid(image, depth):
    """
    Construct the Laplacian pyramid with the desired depth
    Image values are btw [0,1]
    """
    assert image.shape[-1] == 3 # check we have color images
    l_channel = []
    for channel in range(3):
        l_channel.append(Laplacian_pyramid_raw(image[:,:,channel], depth))
    # stacking the three channels to get a list of tensors
    lapl_pyr = [np.stack(i,axis=-1) for i in zip(*l_channel)]
    return lapl_pyr
    

def Collapse_Laplacian_raw(laplacian_pyr):
    """
    Rebuild the image from the Laplacian pyramid for a grey level image
    """
    u = laplacian_pyr[0]
    for i in range(1,len(laplacian_pyr)):
        pad_h = 2*u.shape[0]-laplacian_pyr[i].shape[0]
        pad_w = 2*u.shape[1]-laplacian_pyr[i].shape[1]
        u = laplacian_pyr[i] + Expand(u, pad_h, pad_w)
    return u
    

def Collapse_Laplacian(laplacian_pyr):
    """
    Rebuild the image from the Laplacian pyramid
    """
    l_channel = []
    for channel in range(3):
        one_channel = [ l[:,:,channel] for l in laplacian_pyr]
        l = Collapse_Laplacian_raw(one_channel)
        print(l.shape)
        l_channel.append(Collapse_Laplacian_raw(one_channel))
    # stacking the three channels to get a list of tensors
    u = np.stack(l_channel,axis=-1)
    return u   
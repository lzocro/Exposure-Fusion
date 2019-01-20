#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:04:58 2018

@author: bastien
"""

import numpy as np
import cv2
import warnings
import EF_utils as EF
import measures as m

def EF_main(locations, w_exponents):
    '''
    Main implementation method, combining weights calculation and multiscale merging
    
    '''
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    image = EF.read_sequence_to_fuse(locations)
    
    # weight calculation
    W_norm = m.weight_calc(image, w_exponents) 
    
    # multiscale merging
    
    
    return
    
import numpy as np
import cv2
import warnings


def contrast_measure(image, Max_range = 255., kernel_size = 3, gaussian = False):
    '''
    given an array of images computes the contrast measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    '''
    a,b,c,d = image.shape
    L_im = np.zeros((a,b,c))
    for i in range(image.shape[0]):
        if gaussian == True:
            image[i] = cv2.GaussianBlur(image[i], (kernel_size,kernel_size), 0);#Apply gaussian blur if wanted (might be recomended to denoise the image)
        im_gs = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
        L_im[i] = np.absolute(cv2.Laplacian(im_gs, cv2.CV_32F))
    print('contrast : max is {}, min is {}'.format(np.max(L_im), np.min(L_im)))   
    return L_im


def saturation_measure(image):
    '''
    given an array of images computes the saturation measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    '''     
    r = np.std(image,axis=3)
    print('sat : max is {}, min is {}'.format(np.max(r), np.min(r)))     
    return r


def exposure_measure(image, sigma = 0.2):    
    '''
    given an array of images computes the well-exposedness measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    Set to work with 8bit colour values, might need to refactor it to work with any type.

    Returns a much smaller value than the other two so it's best to multiply it by a big constant like 130
    Maybe we could apply a sigmoid transform instead? what meaning would this have? 
    Or play with the sigma parameter ? 

    '''
    M = np.exp(-((image-0.5)**2)/(2*sigma**2))
    p = np.prod(M,axis=3)
    print('exp : max is {}, min is {}'.format(np.max(p), np.min(p)))
    return np.prod(M,axis=3)


def weight_calc(images, w_exponents, Max_range=255., offset=1):
    '''
    Calculates weight map for each images
    images is a np.array of dims k=nb of images, i,j=image shapes, l=3 (for each RGB color)
    w_exponents is a list of three exponents for each measurement. Each must be comprised between 0 and 1
    
    Returns W : np.array of dims k=nb of images, i,j=image shapes
    '''
#    if (w_exponents.any()<0) | (w_exponents.any()>1):
#        warnings.warn('some exponents are not between 0 and 1', Warning)
    images_norm = np.float32(images)/Max_range
    W = (contrast_measure(images_norm)**w_exponents[0]+offset)*(saturation_measure(images_norm)**w_exponents[1]+offset)*(exposure_measure(images_norm)**w_exponents[2]+offset)
    W += 1e-12 # to avoid dividing by zero
    W_normalized = np.einsum('ij,lij->lij',1./(W.sum(axis=0)),W)
    return np.float32(W_normalized)


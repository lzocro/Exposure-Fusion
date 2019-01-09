import numpy as np
import cv2
import warnings


def contrast_measure(image, Max_range = 255, kernel_size = 3, gaussian = False):
    '''
    given an array of images computes the contrast measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    '''
    a,b,c,d = image.shape
    L_im = np.empty((a,b,c))
    for i in range(image.shape[0]):
        if gaussian == True:
            image[i] = cv2.GaussianBlur(image[i], (kernel_size,kernel_size), 0);#Apply gaussian blur if wanted (might be recomended to denoise the image)
        im_gs = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)/Max_range
        print('nb zero before is {}'.format(np.count_nonzero(im_gs==0)))
        L_im[i] = np.abs(cv2.Laplacian(im_gs, cv2.CV_64F))
        print('nb zero is {}'.format(np.count_nonzero(L_im[i]==0)))
    return L_im


def saturation_measure(image, Max_range = 255):
    '''
    given an array of images computes the saturation measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    '''          
    a,b,c,d = image.shape
    return np.std(image/Max_range,axis=3)


def exposure_measure(image, Max_range = 255, sigma = 0.2, constant= 255/2):    
    '''
    given an array of images computes the well-exposedness measure of Mertens et al.
    returns an array of same dimension as image containing images(arrays) of same size as those in image

    Set to work with 8bit colour values, might need to refactor it to work with any type.
    No need to normalize since we already work with images normalized between 0 and 1

    Returns a much smaller value than the other two so it's best to multiply it by a big constant like 130
    Maybe we could apply a sigmoid transform instead? what meaning would this have? 
    Or play with the sigma parameter ? 

    '''
    a,b,c,d = image.shape
    return np.prod(np.exp(-(image/Max_range-0.5)**2/(2*sigma**2)), axis=3)


def weight_calc(images, w_exponents):
    '''
    Calculates weight map for each images
    images is a np.array of dims k=nb of images, i,j=image shapes, l=3 (for each RGB color)
    w_exponents is a list of three exponents for each measurement. Each must be comprised between 0 and 1
    
    Returns W : np.array of dims k=nb of images, i,j=image shapes
    '''
#    if (w_exponents.any()<0) | (w_exponents.any()>1):
#        warnings.warn('some exponents are not between 0 and 1', Warning)
    W = (contrast_measure(images)**w_exponents[0])*(saturation_measure(images)**w_exponents[1])*(exposure_measure(images)**w_exponents[2])
    W += 1e-12 # to avoid dividing by zero
    W_normalized = np.einsum('ij,lij->lij',1/(W.sum(axis=0)),W)
    return W_normalized

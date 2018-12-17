import cv2
import numpy as np
import scipy

def contrast_measure(image, kernel_size = 3, gaussian = False):
	'''
	given an array of images computes the contrast measure of Mertens et al.
	returns an array of loats of same dimension as image containing images(arrays) of same size as those in image

	'''
	a,b,c,d = image.shape
	L_im = np.empty((a,b,c))
	for i in range(image.shape[0]):
		if gaussian == True:
			image[i] = cv2.GaussianBlur(image[i], (kernel_size,kernel_size), 0)#Apply gaussian blur if wanted (might be recomended to denoise the image)
		im_gs = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
		L_im[i] = cv2.convertScaleAbs(cv2.Laplacian( im_gs, cv2.CV_16S, kernel_size))
	return (L_im)

def saturation_measure(image):
	'''
	given an array of images computes the saturation measure of Mertens et al.
	returns an array of floats same dimension as image containing images(arrays) of same size as those in image

	'''
	a,b,c,d = image.shape
	Sat_im = np.empty((a,b,c))
	Sat_im=np.std(image, axis=-1)
	return(Sat_im)

def exposure_measure(image, Max_range = 255, sigma = 0.2):	
	'''
	given an array of images computes the well-exposedness measure of Mertens et al.
	returns an array of same dimension as image containing images(arrays) of same size as those in image

	Set to work with 8bit colour values, might need to refactor it to work with any type.

	Returns a much smaller value than the other two so it's best to multiply it by a big constant like 130
	Maybe we could apply a sigmoid transform instead? what meaning would this have? 

	'''
	a,b,c,d = image.shape
	Exp_im = np.empty((a,b,c))
	for i in range(image.shape[0]):
		RGB = np.stack(np.array(cv2.split(image[i])), axis=0)
		Exp_im[i] = np.prod(np.exp(-(RGB/Max_range-0.5)**2/(2*sigma)), axis=0)
	return(Exp_im)

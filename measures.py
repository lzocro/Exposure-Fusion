def constrast_measure(image, kernel_size = 3, gaussian = False):
	'''
	given an array of images computes the contrast measure of Mertens et al.
	returns an array of same dimension as image containing images(arrays) of same size as those in image

	'''
	L_im = np.empty(image.shape[0], dtype = 'object')
	t = 0
	for im in image:
		if gaussian == True:
			im = cv2.GaussianBlur(im, (3,3), 0);#Apply gaussian blur if wanted (might be recomended to denoise the image)
		im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		L_im[t] = cv2.convertScaleAbs(cv2.Laplacian( im, cv2.CV_16S, kernel_size))
		t += 1
	return (L_im)


def saturation_measure(image):
	'''
	given an array of images computes the saturation measure of Mertens et al.
	returns an array of same dimension as image containing images(arrays) of same size as those in image

	'''
	Sat_im = np.empty(image.shape[0], dtype = 'object')
	t = 0
	for im in image:
		RGB = np.stack(np.array(cv2.split(im)), axis = 0)
		Sat_im[t] = stats.sem(RGB, axis = 0) #this is weird, a std over 3 values doesn't make any sense?? 
		t += 1
	return(Sat_im)


def exposure_measure(image, Max_range = 255, sigma = 0.2):	
	'''
	given an array of images computes the well-exposedness measure of Mertens et al.
	returns an array of same dimension as image containing images(arrays) of same size as those in image

	Set to work with 8bit colour values, might need to refactor it to work with any type.

	'''
	Exp_im = np.empty(image.shape[0],dtype='object')
	t = 0
	for im in image:
		RGB = np.stack(np.array(cv2.split(im)), axis=0)
		Exp_im[t] = np.prod(np.exp(-(RGB/Max_range-0.5)**2/(2*sigma)),axis=0)
		t += 1
	return(Exp_im)

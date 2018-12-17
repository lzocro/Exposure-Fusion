import cv2
import numpy as np
import scipy
import measures

def read_sequence_to_fuse(seq):

	'''

	seq should be a list of file names such as ["img1.jpg","img2.jpg"]
	The images should have the same dimensions

	'''

	
	n_images = len(seq)
	if n_images == 1:
		warnings.warn('sequence only has one image', Warning)

	height_images = cv2.imread(seq[0]).shape[0]
	width_images = cv2.imread(seq[0]).shape[1]
	output_object = np.empty((n_images, height_images, width_images, 3), dtype=np.uint8)#np.empty((n_images, height_images, width_images, 3)) #3 is for RGB

	for i in range(n_images):
		output_object[i] = cv2.imread(seq[i])

	return(output_object)

def Gaussian_pyramid(image, depth):
	'''
	computes the gaussian pyramid for a given image

	'''
	a = image.shape[0]
	pyr = np.empty(a, dtype='object')
	for i in range(a):
		im = image[i].copy()
		pyr[i] = [im] #adds one original depth
		for _ in range(depth-1):
			im = cv2.pyrDown(im)
			pyr[i].append(im)
	return(pyr)

def Laplacian_pyramid(image, depth):
	'''
	computes the laplacian pyramid for a give image
	(it has depth-1 levels)

	////////

	Attention!! si la profondeur est trop élevée, les dimensions du niveau de la pyramide ne seront plus divisible par deux,
	et l'algorithme echouera, car les dimenstions de G[i][j-1],cv2.pyrUp(G[i][j]) differeront de 1 px. 
	
	A faire: mettre un assert sur depth =< max{n tq height| 2^n or width| 2^n} ' depth -1 ?'

	///////
	'''
	a = image.shape[0]
	pyr = np.empty(a, dtype='object')
	G = Gaussian_pyramid(image, depth+1)
	for i in range(a):
		pyr[i] = []#[G[i][(depth-1)]] #maybe needs to be [] instead
		for j in range((depth-1),0,-1):
			pyr[i].append(cv2.subtract(G[i][j-1],cv2.pyrUp(G[i][j])))
	return(pyr)


def paper_reconstruction(locations, w_exponents, depth):
	'''
	Constructions the second fusion described in section 3.2 give a sequence of images and the exponenets [w_c,w_s,w_e]
	returns a single fused image

	We have a problem when all the weights for all three pictures become 0. What to do? average all three pictures, keep only missing pixels and add to the picture? 

	'''
	assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
	image = read_sequence_to_fuse(locations)
	W = (measures.contrast_measure(image)**w_exponents[0])*(measures.saturation_measure(image)**w_exponents[1])*(measures.exposure_measure(image)**w_exponents[2])
	a,b,c = W.shape
	W_prod = np.empty((a,b,c,3), dtype='float')

	#On copie les poids sur chaque canal couleur
	for i in range(W.shape[0]):	
		W_prod[i] = np.stack((W[i],W[i],W[i]), axis = -1)
	#on normalise les poids
	for i in range(W.shape[0]):
		W_prod[i] = np.multiply( 1/(np.sum(W_prod, axis = 0)), W_prod[i]) #wtilda
	#on prends le laplacien de l'image
	L_I=Laplacian_pyramid(image, depth)
	#on prends la pyramide gaussienne des poids
	G_weights=Gaussian_pyramid(W_prod, depth-1) 
	#on assemble les pyramides
	L_R=[]
	for k in range(depth-1): #loop sur le nb de couches des pyramides
		temp=np.zeros(L_I[0][k].shape) 
		for i in range(image.shape[0]): #loop sur le nd d'images a fusionner
			temp+=np.multiply(G_weights[i][depth-2-k], L_I[i][k]) #somme sur i les produits des pyramides
		L_R.append(temp)
	#on reconstruit la pyramide de R
	R=L_R[0]
	for i in range(1, depth-1):
		R = cv2.pyrUp(R)
		R = cv2.add(R, L_R[i])
	R = np.uint8(R) #convertion de R en uint8 pour RGB 256 
	return(R, W, L_R)

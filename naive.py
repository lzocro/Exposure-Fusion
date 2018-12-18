from EF_utils import read_sequence_to_fuse
import measures as m
import numpy as np
import cv2



def naive_reconstruction(locations, w_exponents):
    '''
    Constructions the naive fusion described in section 3.2 give a sequence of images and the exponenets [w_c,w_s,w_e]
    returns a single fused image

    We have a problem when all the weights for all three pictures become 0. What to do? average all three pictures, keep only missing pixels and add to the picture? 

    '''
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    image = read_sequence_to_fuse(locations)
    W = (m.constrast_measure(image)**w_exponents[0])*(m.saturation_measure(image)**w_exponents[1])*(m.exposure_measure(image)**w_exponents[2])
    a,b,c = W.shape
    print(W.shape)
#    W_prod = np.empty((a,b,c,3), dtype='float')
#    for i in range(W.shape[0]):
#        W[i] = np.multiply( 1/(np.sum(W, axis = 0)), W[i]) #wtilda
#    for i in range(W.shape[0]):    
#        W_prod[i] = np.stack((W[i],W[i],W[i]), axis = -1)
    W_prod = np.einsum('ij,lij->lij',1/(W.sum(axis=0)),W)
    R = np.uint8(np.einsum('lij,lijc->ijc',W_prod, image))
#    R = np.uint8(np.sum(np.multiply(W_prod, image), axis = 0))
    return R, W


# une fonction qui reprends la 1ere formule de 3.2 mais avec quelque améliorations, comme un filtre gaussien sur les poids pour lisser leur effets
def improved_naive_reconstruction(locations, w_exponents):
	'''
	Constructions the naive fusion described in section 3.2 give a sequence of images and the exponenets [w_c,w_s,w_e]
	returns a single fused image
	We have a problem when all the weights for all three pictures become 0. What to do? average all three pictures, keep only missing pixels and add to the picture? 
	'''
	assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
	image = read_sequence_to_fuse(locations)
	W = (m.contrast_measure(image)**w_exponents[0])*(m.saturation_measure(image)**w_exponents[1])*(m.exposure_measure(image)**w_exponents[2])
	a,b,c = W.shape
	W_prod = np.empty((a,b,c,3), dtype='float')
	for i in range(W.shape[0]):
		W[i] = cv2.GaussianBlur(W[i], (3,3), 0) #applying a gaussian filter over the weight map
		W[i] = np.multiply( 1/(np.sum(W, axis = 0)), W[i]) #wtilda (normalisation)
	for i in range(W.shape[0]):	
		W_prod[i] = np.stack((W[i],W[i],W[i]), axis = -1)
	R = np.sum(np.multiply(W_prod, image), axis = 0)
	avg_im=np.mean(image, axis=0)
	Completion= avg_im*(np.uint8(R)<=10)
	R+= Completion #attempt to fill in the holes
	R=np.uint8(R)
	return(R)
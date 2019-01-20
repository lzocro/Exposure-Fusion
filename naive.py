import EF_utils as EF
from measures import weight_calc
from laplacian_pyramid import Gaussian_pyramid, Laplacian_pyramid, Collapse_Laplacian
import numpy as np
import cv2


def naive_reconstruction(locations, w_exponents, Max_range=255., offset=1):
    '''
    Constructions the naive fusion described in section 3.2 give a sequence of images and the exponenets [w_c,w_s,w_e]
    returns a single fused image

    We have a problem when all the weights for all three pictures become 0. What to do? average all three pictures, keep only missing pixels and add to the picture? 

    '''
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    W_norm = weight_calc(image, w_exponents, offset=offset)
    W_norm = np.stack(3*[W_norm], axis=-1)
    R = np.zeros(image[0].shape, dtype=np.float32)
    for i in range(len(image)) :
#        print('im : {}'.format(image[i].shape))
#        print('w : {}'.format(W_norm[i].shape))
        m = cv2.multiply(W_norm[i], image_norm[i], dtype=cv2.CV_32FC3)
        R = cv2.add(R,m)
#    R = np.uint8(np.sum(np.multiply(W_prod, image), axis = 0))
    return R, W_norm

    
def paper_reconstruction(locations, w_exponents, depth=None, Max_range = 255., offset=0):
    """
    Fusion method based on the paper from Mertens & Al
    """
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    # read sequence of images
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    nb_image = len(locations)
    
    # computes weights
    W_norm = weight_calc(image, w_exponents, offset=offset)
    W_norm = np.expand_dims(W_norm, axis=3)
    
    # compute Gaussian pyramid of weights for each image
    list_Gaus_pyr_w = []
    for w in W_norm :
        g, depth = Gaussian_pyramid(w, depth)
        list_Gaus_pyr_w.append(list(reversed((g))))
    list_Gaus_pyr_w = [np.stack(i,axis=0) for i in zip(*list_Gaus_pyr_w)]
    
    # compute Laplacian pyramid of each image
    list_Lapl_pyr = []
    for im in image_norm :
        list_Lapl_pyr.append(Laplacian_pyramid(im, depth))
    lapl_pyr = [np.stack(i,axis=0) for i in zip(*list_Lapl_pyr)]
    
    # fuse Laplacian
    l_lapl_fuse = []
    for l in range(depth):
        R = np.zeros(lapl_pyr[l][0].shape, dtype=np.float32)
        for i in range(nb_image):
            Gauss_stack = np.stack(3*[list_Gaus_pyr_w[l][i][:,:,0]], axis=-1)
            m = cv2.multiply(Gauss_stack, lapl_pyr[l][i], dtype=cv2.CV_32FC3)
            R = cv2.add(R,m)
        l_lapl_fuse.append(R)
    
    # rebuild image
    hdr_im_norm =  Collapse_Laplacian(l_lapl_fuse)
    
    return hdr_im_norm
    
    
    
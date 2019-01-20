
##### Specifics
import cv2
from measures import saturation_measure, contrast_measure
from laplacian_pyramid import Gaussian_pyramid, Laplacian_pyramid, Collapse_Laplacian
import EF_utils as EF
import numpy as np
import scipy.stats as sps

#### Test functions

def exposure_measure(image, sigma = 0.2, distrib='gaus'):
    if distrib == 'gaus':    
        M = np.exp(-((image-0.5)**2)/(2*sigma**2))
    elif distrib == 'laplace':
        M = sps.laplace.pdf(image, loc=0.5, scale=sigma)
    elif distrib == 'beta':
        M = sps.beta.pdf(image,2, 2)
    p = np.prod(M,axis=3)
    print('exp : max is {}, min is {}'.format(np.max(p), np.min(p)))
    return np.prod(M,axis=3)


def weight_calc_test(images, w_exponents, offset=1, Max_range=255., distrib='gaus', sigma=0.2):
    images_norm = np.float32(images)/Max_range
    W = (contrast_measure(images_norm)**w_exponents[0]+offset)*(
            saturation_measure(images_norm)**w_exponents[1]+offset)*(
                    exposure_measure(images_norm, sigma=sigma, distrib=distrib)**w_exponents[2]+offset)
    W += 1e-12 # to avoid dividing by zero
    W_normalized = np.einsum('ij,lij->lij',1./(W.sum(axis=0)),W)
    return np.float32(W_normalized)


def naive_reconstruction_test(locations, w_exponents, offset=1, Max_range=255., distrib='gaus', sigma=0.2):
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    W_norm = weight_calc_test(image, w_exponents, offset=offset, distrib=distrib, sigma=sigma)
    W_norm = np.stack(3*[W_norm], axis=-1)
    R = np.zeros(image[0].shape, dtype=np.float32)
    for i in range(len(image)) :
        m = cv2.multiply(W_norm[i], image_norm[i], dtype=cv2.CV_32FC3)
        R = cv2.add(R,m)
    return R, W_norm



def paper_reconstruction_test(locations, w_exponents, depth=None, offset=1, Max_range = 255., distrib='gaus'):
    """
    Fusion method based on the paper from Mertens & Al
    """
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    # read sequence of images
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    nb_image = len(locations)
    
    # computes weights
    W_norm = weight_calc_test(image, w_exponents, offset=offset, distrib=distrib)
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



##### Run test
    
if __name__ == '__main__' :
    
    exp = [1.,1.,1.]
    location= ['house_A.jpg', 'house_B.jpg', 'house_C.jpg', 'house_D.jpg']
#    location = ['venice_mask_0.jpg', 'venice_mask_1.jpg', 'venice_mask_2.jpg']
    
    distrib = ['gaus', 'laplace', 'beta']
    for d in distrib:
        name_n = './distrib_test/result_image_naive_{}'.format(d)
        name_w = './distrib_test/result_image_naive_weight_{}'.format(d)
        name_p = './distrib_test/result_image_paper_{}'.format(d)
        image_n, W = naive_reconstruction_test(location, exp, distrib=d)
        image_p = paper_reconstruction_test(location, exp, distrib=d)
        
        image_n = np.uint8(np.clip(image_n*255,0,255))
        weight_map = np.uint8(np.clip(W*255,0,255))
        image_p = np.uint8(np.clip(image_p*255,0,255))
        
        # save to file
        cv2.imwrite(name_n+'.jpg', image_n)
        cv2.imwrite(name_p+'.jpg', image_p)
        for i in range(weight_map.shape[0]):
            cv2.imwrite(name_w+'_{}.jpg'.format(i), weight_map[i])
        
        
        # display
#        cv2.namedWindow(name_n,cv2.WINDOW_NORMAL)
#        image_n = cv2.resize(image_n, (600, 800))
#        cv2.imshow(name_n, image_n)
#        cv2.namedWindow(name_w,cv2.WINDOW_NORMAL)
#        weight_map = cv2.resize(weight_map, (600, 800))
#        cv2.imshow(name_p, weight_map)
#        cv2.namedWindow(name_p,cv2.WINDOW_NORMAL)
#        image_n = cv2.resize(image_p, (600, 800))
#        cv2.imshow(name_p, image_p)
    
    
    
    

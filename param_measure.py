
##### Specifics
import cv2
from measures import saturation_measure
from laplacian_pyramid import Gaussian_pyramid, Laplacian_pyramid, Collapse_Laplacian
import EF_utils as EF
import numpy as np

#### Test functions

def contrast_measure(image, Max_range = 255., kernel_size = 3, gaussian = False):
    a,b,c,d = image.shape
    L_im = np.zeros((a,b,c))
    for i in range(image.shape[0]):
        if gaussian == True:
            image[i] = cv2.GaussianBlur(image[i], (kernel_size,kernel_size), 0);#Apply gaussian blur if wanted (might be recomended to denoise the image)
        im_gs = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
        L_im[i] = np.absolute(cv2.Laplacian(im_gs, cv2.CV_32F))
    print('contrast : max is {}, min is {}'.format(np.max(L_im), np.min(L_im)))   
    return L_im

def exposure_measure(image, sigma = 0.2):    
    M = np.exp(-((image-0.5)**2)/(2*sigma**2))
    p = np.prod(M,axis=3)
    print('exp : max is {}, min is {}'.format(np.max(p), np.min(p)))
    return np.prod(M,axis=3)


def weight_calc_test(images, w_exponents, offset=1, Max_range=255., kernel_size=3, sigma=0.2):
    images_norm = np.float32(images)/Max_range
    W = (contrast_measure(images_norm, kernel_size=kernel_size)**w_exponents[0]+offset)*(
            saturation_measure(images_norm)**w_exponents[1]+offset)*(
                    exposure_measure(images_norm, sigma=sigma)**w_exponents[2]+offset)
    W += 1e-12 # to avoid dividing by zero
    W_normalized = np.einsum('ij,lij->lij',1./(W.sum(axis=0)),W)
    return np.float32(W_normalized)


def naive_reconstruction_test(locations, w_exponents, offset=1, Max_range=255., kernel_size=3, sigma=0.2):
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    W_norm = weight_calc_test(image, w_exponents, offset=offset, kernel_size=kernel_size, sigma=sigma)
    W_norm = np.stack(3*[W_norm], axis=-1)
    R = np.zeros(image[0].shape, dtype=np.float32)
    for i in range(len(image)) :
        m = cv2.multiply(W_norm[i], image_norm[i], dtype=cv2.CV_32FC3)
        R = cv2.add(R,m)
    return R, W_norm


def paper_reconstruction_test(locations, w_exponents, depth=None, offset=1, Max_range = 255., kernel_size=3,sigma=0.2):
    """
    Fusion method based on the paper from Mertens & Al
    """
    assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
    # read sequence of images
    image = EF.read_sequence_to_fuse(locations)
    image_norm = np.float32(image/Max_range)
    nb_image = len(locations)
    
    # computes weights
    W_norm = weight_calc_test(image, w_exponents, offset=offset, kernel_size=kernel_size, sigma=sigma)
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
#    location= ['house_A.jpg', 'house_B.jpg', 'house_C.jpg', 'house_D.jpg']
#    name_set = 'house'
    location = ['venice_mask_0.jpg', 'venice_mask_1.jpg', 'venice_mask_2.jpg']
    name_set = 'venice'
    # offset param
    param_range = [0,0.01,0.1,0.5,1,2,10,50,200,1000]
    name = './offset_test'
    for param in param_range:
        name_n = name+'/result_image_naive_{}'.format(param)+name_set
        name_w = name+'/result_image_naive_weight_{}'.format(param)+name_set
        name_p = name+'/result_image_paper_{}'.format(param)+name_set
        image_n, W = naive_reconstruction_test(location, exp, offset=param)
        image_p = paper_reconstruction_test(location, exp, offset=param)
        
        image_n = np.uint8(np.clip(image_n*255,0,255))
        weight_map = np.uint8(np.clip(W*255,0,255))
        image_p = np.uint8(np.clip(image_p*255,0,255))
        
        # save to file
        cv2.imwrite(name_n+'.jpg', image_n)
        cv2.imwrite(name_p+'.jpg', image_p)
        for i in range(weight_map.shape[0]):
            cv2.imwrite(name_w+'_{}.jpg'.format(i), weight_map[i])
        
        # display
#        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
#        image = cv2.resize(image, (600, 600))
#        cv2.imshow(name, image)
        
    # changing kernel for the laplacian
    kernel_size = [1,3,5,10]
    name = './kernel_measure_test'
    for ks in kernel_size:
        name_n = name+'/result_image_naive_{}'.format(ks)+name_set
        name_w = name+'/result_image_naive_weight_{}'.format(ks)+name_set
        name_p = name+'/result_image_paper_{}'.format(ks)+name_set
        image_n, W = naive_reconstruction_test(location, exp, kernel_size=ks)
        image_p = paper_reconstruction_test(location, exp, kernel_size=ks)
        
        image_n = np.uint8(np.clip(image_n*255,0,255))
        weight_map = np.uint8(np.clip(W*255,0,255))
        image_p = np.uint8(np.clip(image_p*255,0,255))
        
        # save to file
        cv2.imwrite(name_n+'.jpg', image_n)
        cv2.imwrite(name_p+'.jpg', image_p)
        for i in range(weight_map.shape[0]):
            cv2.imwrite(name_w+'_{}.jpg'.format(i), weight_map[i])
    
    # changing sigma for the laplacian
    sigma_list = [0.01, 0.1, 0.2, 0.4, 1, 5, 10]
    name = './sigma_test'
    for sigma in sigma_list:
        name_n = name+'/result_image_naive_{}'.format(sigma)+name_set
        name_w = name+'/result_image_naive_weight_{}'.format(sigma)+name_set
        name_p = name+'/result_image_paper_{}'.format(sigma)+name_set
        image_n, W = naive_reconstruction_test(location, exp, sigma=sigma)
        image_p = paper_reconstruction_test(location, exp, sigma=sigma)
        
        image_n = np.uint8(np.clip(image_n*255,0,255))
        weight_map = np.uint8(np.clip(W*255,0,255))
        image_p = np.uint8(np.clip(image_p*255,0,255))
        
        # save to file
        cv2.imwrite(name_n+'.jpg', image_n)
        cv2.imwrite(name_p+'.jpg', image_p)
        for i in range(weight_map.shape[0]):
            cv2.imwrite(name_w+'_{}.jpg'.format(i), weight_map[i])
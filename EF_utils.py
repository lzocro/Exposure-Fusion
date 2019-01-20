import numpy as np
import cv2
import warnings
import measures as m

##### VARIABLES
PATH = './img/'

COMP_SET = {'house' : ['house_A.jpg', 'house_B.jpg', 'house_C.jpg', 'house_D.jpg'],
            'venice' : ['venice_mask_0.jpg', 'venice_mask_1.jpg', 'venice_mask_2.jpg'],
            'ballon' : ['ballon_0{}.png'.format(nb) for nb in np.arange(163,172)],
            'monument' : ['aligned_00{}.png'.format(nb) for nb in np.arange(246,267)],
            
            
            
            }

#### METHODS

def read_sequence_to_fuse(seq):
    '''

    seq should be a list of file names such as ["img1.jpg","img2.jpg"]
    The images should have the same dimensions and be located in the img folder

    '''
    n_images = len(seq)
    if n_images == 1:
        warnings.warn('sequence only has one image', Warning)

    height_images = np.size(cv2.imread(PATH+seq[0]),0)
    width_images = np.size(cv2.imread(PATH+seq[0]),1)
    output_object = np.empty((n_images, height_images, width_images, 3), dtype=np.uint8)#np.empty((n_images, height_images, width_images, 3)) #3 is for RGB

    for i in range(n_images):
        output_object[i] = cv2.imread(PATH+seq[i])

    return output_object


def retrieve_img_seq(name):
    if name not in COMP_SET:
        print('invalid set name')
        return None
    else:
        return COMP_SET[name]
    
       
def save_img(img, path, name, method):
    img = np.uint8(np.clip(img*255,0,255))
    to_save = path+'_'+name+'_'+method+'.jpg'
    cv2.imwrite(to_save, img)
    return img
    
    
    
    
    
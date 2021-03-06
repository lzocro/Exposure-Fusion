import numpy as np
import cv2
import warnings
import measures as m


def read_sequence_to_fuse(seq):
    '''

    seq should be a list of file names such as ["img1.jpg","img2.jpg"]
    The images should have the same dimensions

    '''
    n_images = len(seq)
    if n_images == 1:
        warnings.warn('sequence only has one image', Warning)

    height_images = np.size(cv2.imread(seq[0]),0)
    width_images = np.size(cv2.imread(seq[0]),1)
    output_object = np.empty((n_images, height_images, width_images, 3), dtype=np.uint8)#np.empty((n_images, height_images, width_images, 3)) #3 is for RGB

    for i in range(n_images):
        output_object[i] = cv2.imread(seq[i])

    return output_object



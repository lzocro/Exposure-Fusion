##### Specifics
import cv2
import EF_utils as EF
import numpy as np
from naive import naive_reconstruction, paper_reconstruction

##### Comparison btw methods

def comp_method(img_1, img_2):
    diff = np.abs(img_1-img_2)
    return diff
    

def comp_main(name, exp, depth=None):
    location = EF.retrieve_img_seq(name)
    img_norm = EF.read_sequence_to_fuse(location)
    
    # naive method
    img_naive, W = naive_reconstruction(location, exp)
    
    # Mertens method
    img_paper = paper_reconstruction(location, exp, depth=depth)
    
    # CV2 method
    merge_mertens = cv2.createMergeMertens(contrast_weight=exp[0], saturation_weight=exp[1], exposure_weight=exp[2])
    img_cv2 = merge_mertens.process([img for img in img_norm])

    # save img
    path = './comp_method/'    
    img_naive = EF.save_img(img_naive, path, name, 'naive')
    img_paper = EF.save_img(img_paper, path, name, 'paper')
    img_cv2 = EF.save_img(img_cv2, path, name, 'cv2')
    
    # comp
    img_naive_paper = comp_method(img_naive, img_paper)
    img_naive_paper = EF.save_img(img_naive_paper, path, name, 'naive_vs_paper')
    img_naive_cv2 = comp_method(img_naive, img_cv2)
    img_naive_cv2 = EF.save_img(img_naive_cv2, path, name, 'naive_vs_cv2')
    img_paper_cv2 = comp_method(img_paper, img_cv2)
    img_paper_cv2 = EF.save_img(img_paper_cv2, path, name, 'paper_vs_cv2')
    

##### Main

if __name__ == '__main__' :
    
    exp = [1.,1.,1.]
    for name, l in EF.COMP_SET.items():
        comp_main(name, exp, depth=None)
        
    
    
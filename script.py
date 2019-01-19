##### Python code
import numpy as np
import scipy
##### Param graphiques
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
rc("savefig",**{"jpeg_quality":0.95,'dpi':300})
plt.rcParams["figure.facecolor"]='w'
rc("legend",**{"frameon":True,"shadow":False,'framealpha':0.7,'facecolor':'w'})

##### Nice to have 
from tqdm import tqdm
import time
import timeit

##### Specifics
import cv2
import measures
import EF_utils
import naive


#Read Image works for most formats but NOT RAW
img = cv2.imread('img_forest.jpg')

#example 
img = EF_utils.read_sequence_to_fuse(['img_forest.jpg','img_forest.jpg'])

cv2.namedWindow('title', cv2.WINDOW_NORMAL) #creates a resizeable frame for large pictures, can be skipped
for i in range(len(img)):
    cv2.imshow('title',img[i]) #'title' is title of the window and the id to refer to it (see namedWindow)
    cv2.waitKey(0) #pressing any key continues the program ...
cv2.destroyAllWindows()# ... and closes the window

#example 0
image, W = naive.naive_reconstruction(['house_A.jpg', 'house_B.jpg', 'house_C.jpg', 'house_D.jpg'], [1,1,1])
image = np.uint8(np.clip(image*255,0,255))
cv2.namedWindow('fused image',cv2.WINDOW_NORMAL)
image = cv2.resize(image, (600, 600))
cv2.imshow('fused image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#example 1
image, W = naive.naive_reconstruction(['venice_canal_exp_0.jpg', 'venice_canal_exp_1.jpg', 'venice_canal_exp_2.jpg'], [1.,1.,1.])
image = np.uint8(np.clip(image*255,0,255))
cv2.namedWindow('fused image',cv2.WINDOW_NORMAL)
image = cv2.resize(image, (600, 600))
cv2.imshow('fused image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#example 1
image, W = naive.naive_reconstruction(['venice_mask_0.jpg', 'venice_mask_1.jpg', 'venice_mask_2.jpg'], [1.,1.,1.])
image = np.uint8(np.clip(image*255,0,255))
#cv2.namedWindow('fused image',cv2.WINDOW_NORMAL)
#image = cv2.resize(image, (600, 600))
cv2.imshow('fused image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = naive.paper_reconstruction(['venice_mask_0.jpg', 'venice_mask_1.jpg', 'venice_mask_2.jpg'], [1.,1.,1.])
image = np.uint8(np.clip(image*255,0,255))
#cv2.namedWindow('fused image',cv2.WINDOW_NORMAL)
#image = cv2.resize(image, (600, 600))
cv2.imshow('fused image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# example 2 (paper)
image = naive.paper_reconstruction(['venice_canal_exp_0.jpg', 'venice_canal_exp_1.jpg', 'venice_canal_exp_2.jpg'], [1,1,1], 5)
image = np.uint8(np.clip(image*255,0,255))
cv2.namedWindow('paper image',cv2.WINDOW_NORMAL)
image = cv2.resize(image, (600, 600))
cv2.imshow('paper image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# example 2 (paper)
image = naive.paper_reconstruction(['ballon_0{}.png'.format(nb) for nb in np.arange(163,172)], [1,1,1],5)
image = np.uint8(np.clip(image*255,0,255))
#cv2.namedWindow('paper image',cv2.WINDOW_NORMAL)
#image = cv2.resize(image, (600, 600))
cv2.imshow('paper image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



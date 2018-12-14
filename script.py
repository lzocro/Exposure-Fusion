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

#Read Image works for most formats but NOT RAW
img = cv2.imread('img_forest.jpg')

#Function to import a sequence on pictures into an array
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
	output_object = np.empty(n_images, dtype='object')#np.empty((n_images, height_images, width_images, 3)) #3 is for RGB

	for i in range(len(seq)):
		output_object[i] = cv2.imread(seq[i])

	shapes = [i.shape for i in output_object]
	assert shapes[1:] == shapes[:-1], 'All images do not have the same shape' #checks that all images have the same dimensions
	return(output_object)

#example 
img= read_sequence_to_fuse(['img_forest.jpg','img_forest.jpg'])

cv2.namedWindow('title', cv2.WINDOW_NORMAL) #creates a resizeable frame for large pictures, can be skipped
for i in range(len(img)):
	cv2.imshow('title',img[i]) #'title' is title of the window and the id to refer to it (see namedWindow)
	cv2.waitKey(0) #pressing any key continues the program ...
cv2.destroyAllWindows()# ... and closes the window

#Applying Grayscale filter to image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Saving filtered image to new file
#cv2.imwrite('graytest.jpg',gray)






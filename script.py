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

#Read Image works for most formats but NOT RAW
img = cv2.imread('img_forest.jpg')


#Display Image (in real pixel size, so HD images will be wider than the screen... not sure how to fix )

cv2.namedWindow('title', cv2.WINDOW_NORMAL) #creates a resizeable frame for large pictures, can be skipped
cv2.imshow('title',img) #'title' is title of the window and the id to refer to it (see namedWindow)
cv2.waitKey(0) #pressing any key continues the program ...
cv2.destroyAllWindows()# ... and closes the window

#Applying Grayscale filter to image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Saving filtered image to new file
#cv2.imwrite('graytest.jpg',gray)






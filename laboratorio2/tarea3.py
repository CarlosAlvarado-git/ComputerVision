""" 
Universidad Francisco Marroquin
Computer Vision
Author: Carlos Alvarado - Mario Pisquiy

speedup: 6.255634888000573 seconds
sharp: 114.21909189224243 seconds
fast-sharp: 18.25859308242798 seconds
"""
import sys ; sys.path.append("../")
import cv2 as cv
import numpy as np
from external import sharpen_cython
import matplotlib.pyplot as plt
import time
# get camera handle 
device_id = 0
# cap = cv.VideoCapture(device_id)
cap = cv.VideoCapture("./pexels_videos_2431225 (240p).mp4")
# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")
start_time = time.time()
while True:
    try:
        ret, im_rgb = cap.read()
        im = im_rgb[:,:,:]
        if ret:
            # apply operation
            kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
            sharpen_image = np.asarray(sharpen_cython(img=im, kernel=kernel)) #sharpen de c y convertir el resultado a un numpy
            # # create windows
            win0 = 'Original'
            win1 = 'Processed'

            r,c = im.shape[0:2]
            resize_factor = 2

            R = int(r//resize_factor)
            C = int(c//resize_factor)
            win_size = (C, R) 

            cv.namedWindow(win0, cv.WINDOW_NORMAL)
            cv.namedWindow(win1, cv.WINDOW_NORMAL)

            cv.resizeWindow(win0, (win_size[0]//2,win_size[1]//2))
            cv.resizeWindow(win1, win_size)

            cv.imshow(win0, im)
            cv.imshow(win1, sharpen_image)
        
            # # align windows        
            cv.moveWindow(win1, 0, 0)
            cv.moveWindow(win0, C, 0)
            
            # # exit with q
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    except:
        break
#clean up before exit
cap.release()
cv.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))

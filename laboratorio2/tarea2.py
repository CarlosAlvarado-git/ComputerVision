""" 
Universidad Francisco Marroquin
Computer Vision
Author: Carlos Alvarado - Mario Pisquiy
"""
import sys ; sys.path.append("../")
import cv2 as cv
import numpy as np

# get camera handle 
device_id = 0
cap = cv.VideoCapture(device_id)

# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")

# set video capture properties
# hardware specific

#cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024);cap.set(cv.CAP_PROP_FRAME_HEIGHT, 576)
#cap.set(cv.CAP_PROP_FRAME_WIDTH, 800);cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
#sharp effect
def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv.filter2D(img, -1, kernel)
    return img_sharpen

# get frame, apply processing and show result
while True:
    ret, im_rgb = cap.read()
    im = im_rgb[:,:,:]

    if ret:
        # apply operation
        sharpen_image = sharpen(im)

        
        # create windows
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
	
        # align windows        
        cv.moveWindow(win1, 0, 0)
        cv.moveWindow(win0, C, 0)
        
        # exit with q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#clean up before exit
cap.release()
cv.destroyAllWindows()

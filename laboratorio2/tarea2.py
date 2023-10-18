""" 
Universidad Francisco Marroquin
Computer Vision
Author: Carlos Alvarado - Mario Pisquiy
"""
import sys ; sys.path.append("../")
import cv2 as cv
import numpy as np
import time
# get camera handle 
device_id = 0
# cap = cv.VideoCapture(device_id)
cap = cv.VideoCapture("./pexels_videos_2431225 (240p).mp4")
# verify that video handle is open
if (cap.isOpened() == False):
    print("Video capture failed to open")

start_time = time.time()
def sharpen(img:np.array, kernel:np.array):
    
    # Get the height, width, and number of channels of the image
    height,width,c =img.shape[0],img.shape[1],img.shape[2]
    
    # Get the height, width, and number of channels of the kernel
    kernel_height,kernel_width = kernel.shape[0],kernel.shape[1]
    
    # Create a new image of original img size minus the border 
    # where the convolution can't be applied
    new_img = np.zeros((height-kernel_height+1,width-kernel_width+1,3)) 
    
    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(kernel_height//2, height-kernel_height//2-1):
        for j in range(kernel_width//2, width-kernel_width//2-1):
            # Extract a window of pixels around the current pixel
            window = img[i-kernel_height//2 : i+kernel_height//2+1,j-kernel_width//2 : j+kernel_width//2+1]
            # Apply the convolution to the window and set the result as the value of the current pixel in the new image
            var = np.absolute(int((window[:,:,0] * kernel).sum()))
            if (var > 255):
                new_img[i, j, 0] = int(255)
            else:
                new_img[i, j, 0] = var
            var = np.absolute(int((window[:,:,1] * kernel).sum()))
            if (var > 255):
                new_img[i, j, 1] = int(255)
            else:
                new_img[i, j, 1] = var
            var = np.absolute(int((window[:,:,2] * kernel).sum()))
            if (var > 255):
                new_img[i, j, 2] = int(255)
            else:
                new_img[i, j, 2] = var
      
    # Clip values to the range 0-255
    #new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)

# get frame, apply processing and show result
while True:
    try:
        ret, im_rgb = cap.read()
        im = im_rgb[:,:,:]

        if ret:
            # apply operation
            kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
            sharpen_image = sharpen(img=im, kernel=kernel)
            
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
    except:
        break

#clean up before exit
cap.release()
cv.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))
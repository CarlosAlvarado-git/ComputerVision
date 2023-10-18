import cython
import numpy as np
@cython.boundscheck(False)
cpdef unsigned char[:, :,:] sharpen_cython(unsigned char[:, :,:] img, double[:, :] kernel):
    cdef int kernel_height, kernel_width, x, y, R, height, width, c, i , j, t, z
    # cdef unsigned char[:, :,] new_img
    # Get the height, width, and number of channels of the image
    height,width,c =img.shape[0],img.shape[1],img.shape[2]
    
    # Get the height, width, and number of channels of the kernel
    kernel_height,kernel_width = kernel.shape[0],kernel.shape[1]
    
    # Create a new image of original img size minus the border 
    # where the convolution can't be applied
    #cdef np.ndarray[FTYPE_t, ndim=2, mode='c'] new_img
    new_img = np.zeros((height-kernel_height+1,width-kernel_width+1,3)) 
    R = kernel_height//2
    # Loop through each pixel in the image
    # But skip the outer edges of the image
    for i in range(R, height-R-1):
        for j in range(R, width-R-1):
            # Extract a window of pixels around the current pixel
            window = img[i-R : i+R+1,j-R: j+R+1]
            # Apply the convolution to the window and set the result as the value of the current pixel in the new image
            for z in range(c):
                t = 0
                for x in range(kernel_height):
                    for y in range(kernel_width):
                            t += int(window[x, y,z] * kernel[x, y])
                #t = t if t>0 else (t*-1)
                t = abs((t))
                if (t > 255):
                    t = 255
                new_img[i,j,z] = t
    return new_img.astype(np.uint8)
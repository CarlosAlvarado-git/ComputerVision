import cv2 as cv
import numpy as np
from external import sharpen_cython

img_sharpen = sharpen_cython(img=img.copy(), kernel=kernel)
cvlib.imgcmp(img, img_sharpen, ["Normal", "Sharpen"])
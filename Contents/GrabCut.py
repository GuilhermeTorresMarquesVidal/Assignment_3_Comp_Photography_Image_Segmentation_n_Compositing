import numpy as np
import cv2 as cv

def rect_GrabCut(image, rect_final):

    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    x,y,w,h = rect_final 
    mask[y:y+h, x:x+w] = 1

    mask, bgdModel, fgdModel = cv.grabCut(
        image, mask, rect_final,
        bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1)
    image_cut = image * mask2[:,:,np.newaxis]

    return image_cut
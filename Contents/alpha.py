import numpy as np
from skimage.transform import resize

def alpha_composition(source, destiny, mask):

    objects = source*mask
    background = destiny*(1.0-mask)
    result  = objects + background

    return result

def resize_and_composition(source, destiny, mask):

    size = destiny.shape

    source_resize = resize(source, size, anti_aliasing = True)
    mask_resize = resize(mask, size, anti_aliasing = True)

    result = alpha_composition(source = source_resize, destiny = destiny, mask =  mask_resize)

    return result
import numpy as np

def generate_mask(image, kind):

    mask = np.zeros_like(image)
    mask = mask.astype(np.float64)
    size = mask.shape
    half_x = int(size[1]/2)
    half_y = int(size[0]/2)

    if(kind == 'standing'):
        mask[:,half_y:,:] = 1.0
    elif(kind == 'flat'):
        mask[half_x:,:,:] = 1.0
    else:
        pass

    return mask
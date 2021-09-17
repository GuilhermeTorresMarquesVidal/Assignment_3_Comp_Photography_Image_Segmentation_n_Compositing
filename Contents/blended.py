import numpy as np
from scipy.ndimage import zoom

from .params import zoom_out

def laplacian_out(image_A_seq, image_B_seq, mask_seq):

    if(len(image_A_seq) == 0 or len(image_B_seq) == 0 or len(mask_seq) == 0):
        return [], []
    
    laplacian_out_alpha, laplacian_out_beta = laplacian_out(
            image_A_seq = image_A_seq[1:], 
            image_B_seq = image_B_seq[1:], 
            mask_seq = mask_seq[1:])

    image_A = image_A_seq[0]
    image_B = image_B_seq[0]
    mask = mask_seq[0]

    alpha = mask*image_A + (1.0 - mask)*image_B
    beta = mask*image_B + (1.0 - mask)*image_A

    laplacian_out_alpha.insert(0, alpha)
    laplacian_out_beta.insert(0, beta)

    return laplacian_out_alpha, laplacian_out_beta

def blended(laplacian_out):

    if(len(laplacian_out) == 1):
        return laplacian_out[0]

    image_result_i = blended(laplacian_out[1:])

    image_i = laplacian_out[0]

    image_zoom_out = np.zeros_like(image_i)

    # Zoom operation

    image_zoom_out[:,:,0] = zoom(image_result_i[:,:,0], zoom_out)
    image_zoom_out[:,:,1] = zoom(image_result_i[:,:,1], zoom_out)
    image_zoom_out[:,:,2] = zoom(image_result_i[:,:,2], zoom_out)

    image_result = image_i + image_zoom_out

    return image_result

def blended_pipeline(image_A_seq, image_B_seq, mask_seq):

    laplacian_out_alpha, laplacian_out_beta = laplacian_out(
        image_A_seq = image_A_seq, 
        image_B_seq = image_B_seq, 
        mask_seq = mask_seq)

    image_result_alpha = blended(laplacian_out = laplacian_out_alpha)
    image_result_beta = blended(laplacian_out = laplacian_out_beta)

    return image_result_alpha, image_result_beta

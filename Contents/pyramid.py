import numpy as np 
from scipy.ndimage import zoom, gaussian_filter

from .params import sigma, zoom_in, zoom_out

def generate(image, dep_max, layer):

    if(layer == dep_max):
        return [image], [image]

    # Blur process

    image_blur = np.zeros_like(image)

    image_blur[:,:,0] = gaussian_filter(image[:,:,0], sigma = sigma)
    image_blur[:,:,1] = gaussian_filter(image[:,:,1], sigma = sigma)
    image_blur[:,:,2] = gaussian_filter(image[:,:,2], sigma = sigma)

    # Resize image

    size = image_blur.shape

    image_zoom_in = np.zeros((int(size[0]/2), int(size[1]/2), size[2]))

    image_zoom_in[:,:,0] = zoom(image_blur[:,:,0], zoom_in)
    image_zoom_in[:,:,1] = zoom(image_blur[:,:,1], zoom_in)
    image_zoom_in[:,:,2] = zoom(image_blur[:,:,2], zoom_in)

    gauss_seq, laplace_seq = generate(image = image_zoom_in, dep_max = dep_max, layer = layer + 1)

    # create recovery map

    image_zoom_out = np.zeros_like(image)

    image_zoom_out[:,:,0] = zoom(gauss_seq[0][:,:,0], zoom_out)
    image_zoom_out[:,:,1] = zoom(gauss_seq[0][:,:,1], zoom_out)
    image_zoom_out[:,:,2] = zoom(gauss_seq[0][:,:,2], zoom_out)

    image_diff = image - image_zoom_out

    # insert image and diference in the image

    gauss_seq.insert(0, image)
    laplace_seq.insert(0, image_diff)

    return gauss_seq, laplace_seq



def generate_pyramid(image, dep_max):

    return generate(image = image, dep_max = dep_max, layer = 0)
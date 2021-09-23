import numpy as np

from Contents.pyramid import generate_pyramid
from Contents.blended import blended_pipeline

def pyramid_collection(images, masks_images, dep_max_list):

    apple_collection = dict()
    orange_collection = dict()
    masks_collection = dict()

    for dep_max in dep_max_list:

        apple = dict()
        orange = dict()
        masks = dict()
        masks['standing'] = dict()
        masks['flat'] = dict()

        apple['gauss'], apple['laplace'] = generate_pyramid(
            image = images['apple'], dep_max = dep_max)

        orange['gauss'], orange['laplace'] = generate_pyramid(
            image = images['orange'], dep_max = dep_max)

        masks['standing']['gauss'], masks['standing']['laplace'] = generate_pyramid(
            image = masks_images['standing'], dep_max = dep_max)

        masks['flat']['gauss'], masks['flat']['laplace'] = generate_pyramid(
            image = masks_images['flat'], dep_max = dep_max)

        apple_collection[dep_max] = apple
        orange_collection[dep_max] = orange
        masks_collection[dep_max] = masks

    return apple_collection, orange_collection, masks_collection

def blended_collection(apple_collection, orange_collection, masks_collection, dep_max_list):

    image_result_alpha_collection = dict()
    image_result_beta_collection = dict()

    for dep_max in dep_max_list:

        image_result_alpha = dict()
        image_result_beta = dict()

        image_result_alpha['standing'], image_result_beta['standing'] = blended_pipeline(
            image_A_seq = apple_collection[dep_max]['laplace'], 
            image_B_seq = orange_collection[dep_max]['laplace'], 
            mask_seq = masks_collection[dep_max]['standing']['gauss'])

        image_result_alpha['flat'], image_result_beta['flat'] = blended_pipeline(
            image_A_seq = apple_collection[dep_max]['laplace'], 
            image_B_seq = orange_collection[dep_max]['laplace'], 
            mask_seq = masks_collection[dep_max]['flat']['gauss'])

        image_result_alpha_collection[dep_max] = image_result_alpha
        image_result_beta_collection[dep_max] = image_result_beta

    return image_result_alpha_collection, image_result_beta_collection
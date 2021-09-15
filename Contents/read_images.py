import matplotlib.image as mpimg

def read_image(filepath):

    return mpimg.imread(filepath)

def read_list_of_images(list_of_files, keys):

    images = {}

    for filepath, key in zip(list_of_files, keys):

        images[key] = read_image(filepath = filepath)

    return images
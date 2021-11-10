__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from PIL import Image
import numpy as np

def load_images(address_list,image_size):
    '''
    Load images as np.array based on a list of address. 

    Arguments:
    address_list -- the list of image address to load from. 
    image_size -- the size of squared images for output

    Returns:
    images -- a ndarray storing the images with shape [n_image, image_size, image_size, 3]
    '''
    image_list = []

    for image_address in address_list:
        #load image
        image = Image.open(image_address)

        #resize and crop image
        shape = np.array(image.size)
        short_side = min(shape)
        new_shape = (shape/short_side*image_size).astype(int)

        left = int(max(new_shape)/2-image_size/2)
        right = int(max(new_shape)/2+image_size/2)
        upper = 0
        lower = image_size
        box = [left, upper, right, lower]

        image = image.resize(new_shape)
        image = image.crop(box)       
        image = image.resize([image_size,image_size])

        image_list.append(np.array(image))
    
    images = np.stack(image_list)

    return(images)

def split_list(l, index):
    '''
    Split the list into two parts using the index.

    Arguments:
    l -- the list.
    index -- index where the split happens.

    return:
    list_1, list2
    '''
    return l[:index], l[index:]
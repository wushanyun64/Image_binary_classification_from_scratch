__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from PIL import Image
import numpy as np
import h5py

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

def load_data(address_train, address_test):
    '''
    Load Cat or Not Cat dataset. 
    '''
    train_dataset = h5py.File(address_train, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(address_test, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

import numpy as np
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    By: https://github.com/Bjarten
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

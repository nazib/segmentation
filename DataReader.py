import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import glob
import os
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tqdm import tqdm
import PIL


def load_dataset(path,IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=3):

    filenames = glob.glob(os.path.join(path,'tiles','*.jpg'))
    labels = glob.glob(os.path.join(path,'annotations','*.png'))
    X_train = np.zeros((len(filenames), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(labels), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.float32)

    for n, (img_name,label_name) in tqdm(enumerate(zip(filenames,labels)), total=len(filenames)):
    
        #Read image files iteratively
        img = Image.open(img_name)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH),resample=PIL.Image.BILINEAR)
        img = np.asarray(img)        
        #Append image to numpy array for train dataset
        X_train[n] = img
        
        #Read corresponding mask files iteratively
        mask = Image.open(label_name).convert('L')
        mask = mask.resize((IMG_HEIGHT, IMG_WIDTH),resample=Image.NEAREST)
        mask = np.asarray(mask)    
        mask[mask>0] =1.0
        mask = np.expand_dims(mask,axis=2)
        Y_train[n] = mask

    return X_train,Y_train

        

import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
from model import*
from DataReader import*
from matplotlib import pyplot as plt
import datetime
from tensorflow.keras.models import load_model

def dice_metric(seg,gt):
    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def predict(X,Y):
    model = load_model('unet_12_02_2022.h5',compile=False)
    preds = model.predict(X, verbose=1)
    print(len(preds))
    all_dice =[]
    if not os.path.exists('predictions'):
        os.mkdir('predictions')
    
    for i,(seg,gt) in enumerate(zip(preds,Y)):
        all_dice.append(dice_metric(seg,gt))
        seg = np.squeeze(seg)
        seg = seg.astype(np.uint8)
        seg = Image.fromarray(seg)
        seg = seg.resize((512, 512),resample=Image.NEAREST)
        fname = f'predict_{i}.tiff'
        seg.save(os.path.join('predictions',fname))

        

if __name__ == "__main__":
    TRAIN_PATH = r'C:\DM\MLops\algea\micrograph_data'
    #TEST_PATH = '../input/stage1_test/'
    dataset_im, dataset_label = load_dataset(TRAIN_PATH)
    split = 0.2
    train_len = int(len(dataset_im)*(1.0-0.2))
    train_x = dataset_im[:train_len]
    train_y = dataset_label[:train_len]
    #trainer(train_x,train_y)
    predict(dataset_im[train_len:],dataset_label[train_len:])
    
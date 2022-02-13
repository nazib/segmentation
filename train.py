import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
from model import*
from DataReader import*
from matplotlib import pyplot as plt
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import logging
import sys


def trainer(data_dir,epochs,size,batch_size,lr,val_percent):
    ## Defining network architecture
    image_size, n_class = [size,size,3], 1
    structure = [64,128,256,512] 
    model = Unet(image_size,n_class,structure,lr).createModel()
    model.summary()
    print("Unet Model Created")
    
    ## Loading data
    data_im,data_label = load_dataset(data_dir)
    train_len = int(len(data_im)*(1.0-val_percent))
    train_im = data_im[:train_len]
    train_label = data_label[:train_len]

    earlystopper = EarlyStopping(patience=15, verbose=1)
    today = datetime.datetime.now()
    model_name = f'unet_{today.strftime("%d")}_{today.strftime("%m")}_{today.strftime("%Y")}.h5'
    checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
    
    results = model.fit(train_im, train_label, validation_split=0.1, batch_size=batch_size, epochs=epochs, 
                    callbacks=[earlystopper, checkpointer])
    print("Traning Complete")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-dim', '--dimension',metavar='Dim', type=int, default=128,
                        help='Input Image Dimension', dest='dimension')

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-d', '--dir', dest='dir', type=str, default=r"C:\DM\MLops\algea\micrograph_data",
                        help='Training Data Directory')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()
    try:
        trainer(  data_dir = args.dir,
                  epochs=args.epochs,
                  size = args.dimension,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        print('Training Inturrepted')
        #logging.info('Training Inturrepted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




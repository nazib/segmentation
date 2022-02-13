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
import argparse
import sys

def dice_metric(seg,gt):
    dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def predict(data_dir,image_dim,model_dir,with_label):
    ## Loading Trained model ##
    try:
        models = glob.glob(os.path.join(model_dir,'*.h5'))
        latest_model = max(models, key=os.path.getctime)
        model = load_model(latest_model,compile=False)
    except:
         print("Problems in model loading. Check if model exists")
         sys.exit(0)

    ### Saving the predicted images and their corresponding Dice Accuracy with the GT annotations##
    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    if with_label is True:
        test_im,test_labels = load_dataset(data_dir)
        all_dice =[]
        preds = model.predict(test_im, verbose=1)

        for i,(seg,gt) in enumerate(zip(preds,test_labels)):
            all_dice.append(dice_metric(seg,gt))
            seg = np.squeeze(seg)
            seg = seg.astype(np.uint8)
            seg = Image.fromarray(seg)
            seg = seg.resize((512, 512),resample=Image.NEAREST)
            fname = f'predict_{i}.tiff'
            seg.save(os.path.join('predictions',fname))
        
        fname = os.path.join("predictions","Dice.txt")
        acc_file = open(fname,"w")
        acc_file.write(np.mean(all_dice))
        return "Predicted successfully.Please Check Prediction Directory"
    else:
        test_im = load_dataset(data_dir,isTest=True)
        preds = model.predict(test_im, verbose=1)
        for i,seg in enumerate(preds):
            seg = np.squeeze(seg)
            seg = seg*255.0
            seg = seg.astype(np.uint8)
            seg = Image.fromarray(seg)
            seg = seg.resize((512, 512),resample=Image.NEAREST)
            fname = f'predict_{i}.tiff'
            seg.save(os.path.join('predictions',fname))
        return "Predicted successfully.Please Check Prediction Directory"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dim', '--dimension',metavar='Dim', type=int, default=128,
                        help='Input Image Dimension', dest='dimension')
    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, default=r"C:\DM\MLops\algea\test",
                        help='Training Data Directory')
    parser.add_argument('-md', '--model_dir', dest='model_dir', type=str, default="chkpoints",
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-wl', '--with_label', dest='with_label', type=bool, default=True,
                        help='To indicate test dataset has Ground Truth labels or not')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    try:
        predict( data_dir = args.data_dir,
                 image_dim = args.dimension,
                 model_dir= args.model_dir,
                 with_label = args.with_label)
    except KeyboardInterrupt:
        print('Prediction Inturrepted')
        #logging.info('Training Inturrepted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    
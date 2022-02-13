import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam,RMSprop, SGD
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
from keras import backend as K
from losses import*


class Unet:
    def __init__(self,im_size,n_class,filters,lr):
        self.im_size=im_size
        self.n_class = n_class
        self.n_filters =filters
        self.Lr=lr

    def myConv(self,x_in, nf, strides=1):
        x_out = tf.keras.layers.Conv2D(nf,kernel_size=3, padding='same',kernel_initializer='he_normal',strides=strides)(x_in)
        x_out = tf.keras.layers.LeakyReLU(0.2)(x_out)
        x_out = tf.keras.layers.BatchNormalization()(x_out)
        x_out = tf.keras.layers.Dropout(0.1) (x_out)

        return x_out

    def Decoder(self):
        x = self.myConv(self.Encoder_layer3,self.n_filters[-1])
        x = tf.keras.layers.concatenate([x, self.Encoder_layer3])
        x = tf.keras.layers.UpSampling2D()(x)
        x = self.myConv(x,self.n_filters[-2])
        x = tf.keras.layers.concatenate([x, self.Encoder_layer2])
        x = self.myConv(x, self.n_filters[-3])
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.concatenate([x, self.Encoder_layer1])
        x = self.myConv(x, self.n_filters[-4])
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.concatenate([x, self.Encoder_layer0])
        x = self.myConv(x, 2)
        x = tf.keras.layers.UpSampling2D()(x)
        seg = tf.keras.layers.Conv2D(self.n_class, kernel_size=1,activation='sigmoid', name='seg')(x)
        return  seg

    def createModel(self):

        self.inp = tf.keras.Input(shape=self.im_size)
        
        ### Instead of using maxpool I decided to use strided (stride=2) convolution
        ### Strided convolution gives more weight parameters than maxpool, hence performance improves  
        self.Encoder_layer0 = self.myConv(self.inp, self.n_filters[0], 2)
        self.Encoder_layer1 = self.myConv(self.Encoder_layer0,self.n_filters[1], 2)
        self.Encoder_layer2 = self.myConv(self.Encoder_layer1,self.n_filters[2], 2)
        self.Encoder_layer3 = self.myConv(self.Encoder_layer2,self.n_filters[3], 2)

        segmentation  = self.Decoder()
        network = Model(inputs=self.inp, outputs=segmentation)
        network.compile(loss=binary_crossentropy,
                        optimizer=Adam(self.Lr, beta_1=.5, beta_2=0.9),
                        metrics=['accuracy'])
        return network
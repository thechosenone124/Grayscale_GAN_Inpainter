import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model

from pconv_layer import PConv2D
# source: https://github.com/MathiasGruber/PConv-Keras

class PConvUnet(object):
        
    def __init__(self, img_rows=512, img_cols=512, weight_filepath=None):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None
        
        Args:
            img_rows (int): image height.
            img_cols (int): image width.
            weight_filepath: filepath containing weights"""
        
        # Settings
        self.weight_filepath = weight_filepath
        self.img_rows = img_rows
        self.img_cols = img_cols
        assert self.img_rows >= 256, 'Height must be >256 pixels'
        assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0
        
        # Create UNet-like model
        self.input, self.output, self.model = self.build_pconv_unet()
        
    #Modified for grayscale images and masks
    def build_pconv_unet(self, train_bn=True):      

        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bottleneck=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bottleneck: #UNET architecture: a skip connection so low level data doesnt have to pass through whole network
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bottleneck=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
        
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bottleneck=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bottleneck: #UNET architecture: a skip connection so low level data doesnt have to pass through whole network
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
            
        d_conv8, d_mask8 = decoder_layer(e_conv7, e_mask7, e_conv6, e_mask6, 512, 3)
        d_conv9, d_mask9 = decoder_layer(d_conv8, d_mask8, e_conv5, e_mask5, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv4, e_mask4, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv3, e_mask3, 256, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv2, e_mask2, 128, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv1, e_mask1, 64, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, inputs_img, inputs_mask, 3, 3, bottleneck=False)
        outputs = Conv2D(1, 1, activation = 'sigmoid', name='outputs_img')(d_conv16)
        
        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

        return [inputs_img, inputs_mask], outputs, model   

    #Do not run this. The generator has no loss function and should not be compiled    
    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer = Adam(lr=lr),
            #We don't need this, we have discriminator loss
            #loss=self.loss_total(inputs_mask),
            metrics=[self.PSNR]
        )
    
    #This is for training the generator the conventional way, we have our GANN training
    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )
        
    def predict(self, sample):
        """Run prediction using this model"""
        return self.model.predict(sample)
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())
    
    def save(self):        
        self.model.save_weights(self.current_weightfile())
         
    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model, inputs_mask = self.build_pconv_unet(train_bn)
        self.compile_pconv_unet(self.model, inputs_mask, lr) 

        # Load weights into model
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)

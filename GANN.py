'''
Author: Bryan Tan
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 TBD.py
'''

import numpy as np
import time
import gc
import sys
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Input, Concatenate
from keras.layers import LeakyReLU, Dropout, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from pconv_model import PConvUnet
import matplotlib.pyplot as plt

from MaskGenerator import MaskGenerator

#SETTINGS

TRAIN_DIR = "TrainingImages/512px/train"
VAL_DIR = "TrainingImages/512px/val"
TEST_DIR = "TrainingImages/512px/test"
BATCH_SIZE = 10
MODEL_FILEPATH = "AM.h5"
NUM_EPOCHS=20
class AugmentingDataGenerator(ImageDataGenerator):
    #Keras' ImageDataGenerator's flow from directory generates batches of augmented images
    #We need masks and images together, so this wraps ImageDataGenerator
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)        
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample            
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            # Because we want to generate what is here, so we dont want the model to know
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([masked_image, mask],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class DCGAN(object):
    def __init__(self, img_rows=512, img_cols=512, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.gen_input = None
        self.gen_output = None
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.generator_network = PConvUnet(img_rows, img_cols)
   
    def my_usual_layer(self, depth, img, pooling=True):
        out = Conv2D(depth, 5, strides=2, padding='same')(img)
        out1 = LeakyReLU(alpha=0.2)(out)
        out2 = BatchNormalization()(out1)
        out3 = MaxPooling2D((2,2))(out2)
        if pooling:
            return out3
        return out2  
        
    # Takes in an image and a mask, and determines if the area under the mask is real or fake
    def discriminator(self):
        if self.D:
            return self.D
        masked_image = Input((self.img_rows, self.img_cols, 1))
        mask = Input((self.img_rows, self.img_cols, 1))
        filled_img = Input((self.img_rows, self.img_cols, 1))
        
        combined = Concatenate(axis=3)([masked_image, mask, filled_img]) # Overlay mask and filled image
        
        depth = 32
        combined1 = self.my_usual_layer(depth, combined)
        combined2 = self.my_usual_layer(depth*2, combined1)
        combined3 =  self.my_usual_layer(depth, combined2, pooling=False)

        flattened = Flatten()(combined3)
        dense1 = Dense(32)(flattened)
        output = Dense(1, activation='sigmoid')(dense1)

        self.D = Model(inputs=[masked_image, mask, filled_img], outputs=output)
        print(self.D.summary())
        return self.D

    #Partial convolutional generator
    def build_generator(self):
        if self.G:
            return
            
        self.G = self.generator_network
        self.gen_input = self.G.input
        self.gen_output = self.G.output

    def compile_discriminator(self, trainable=True):
        self.DM.trainable = trainable 
        optimizer = RMSprop(lr=0.001)
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])

    def adversarial_model(self, file=None):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.01, decay=9e-5)
        #build discriminator and generator
        self.build_generator()
        self.DM = self.discriminator()
        self.AM = Model(self.gen_input, self.DM([self.gen_input[0], self.gen_input[1], self.gen_output]))
        
        if file is not None:
            self.load_adversarial_weights(file)
            
        self.compile_discriminator() # Discriminator is compiled trainable
        self.DM.trainable = False # Adversarial model is for generator training
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def save_adversarial_weights(self, filepath=MODEL_FILEPATH):
        self.AM.save_weights(filepath)

    def load_adversarial_weights(self, filepath):
        self.AM.load_weights(filepath)

# Nothing to do with MNIST
class MNIST_DCGAN(object):
    def __init__(self, file=None):
        # Set target image size here
        self.img_rows = 256
        self.img_cols = 256
        self.channel = 1
        #set up input
        # Create training generator
        train_datagen = AugmentingDataGenerator(  
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            horizontal_flip=True
        )
        self.train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR, 
            MaskGenerator(self.img_rows, self.img_cols, 1),
            target_size=(self.img_rows, self.img_cols), 
            batch_size=BATCH_SIZE,
            color_mode="grayscale"
        )

        # Create validation generator
        no_aug = AugmentingDataGenerator(rescale=1./255)
        self.val_generator = no_aug.flow_from_directory(
            VAL_DIR, 
            MaskGenerator(self.img_rows, self.img_cols, 1),
            target_size=(self.img_rows, self.img_cols), 
            batch_size=BATCH_SIZE,
            seed=42,
            color_mode="grayscale"
        )

        # Create testing generator
        self.test_generator = no_aug.flow_from_directory(
            TEST_DIR, 
            MaskGenerator(self.img_rows, self.img_cols, 1),
            target_size=(self.img_rows, self.img_cols), 
            batch_size=10, 
            seed=42,
            color_mode="grayscale"
        )
        #Display some sample images
        # Pick out an example
        # test_data = next(self.test_generator)
        # (masked, mask), ori = test_data

        # # Show side by side
        # for i in range(len(ori)):
            # _, axes = plt.subplots(1, 3, figsize=(20, 5))
            # axes[0].imshow(masked[i,:,:,0])
            # axes[1].imshow(mask[i,:,:,0] * 1.)
            # axes[2].imshow(ori[i,:,:,0])
            # plt.show()

        self.DCGAN = DCGAN(self.img_rows, self.img_cols)
        self.adversarial = self.DCGAN.adversarial_model(file)
        self.discriminator = self.DCGAN.DM
        self.generator = self.DCGAN.G

    def plot_callback(self, model):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""

        test_data = next(self.test_generator)
        (masked, mask), ori = test_data

        # Get samples & Display them        
        pred_img = model.predict([masked, mask])
        # pred_img2 = model.predict([masked, mask])
        # plt.set_cmap('gray')
        # Show side by side
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,0])
            axes[1].imshow(mask[i,:,:,0] * 1.)
            axes[2].imshow(ori[i,:,:,0])
            axes[3].imshow(pred_img[i,:,:,0] * 1.)
            # axes[4].imshow(pred_img2[i,:,:,0] * 1.)
            plt.show()
                    
            #plt.savefig(r'data/test_samples/img_{}.png'.format(i))
            plt.close()
    
    # One epoch
    # Totally ignore train steps, woopsie
    def train(self, train_steps=None, val_steps=25, save_interval=0):
        step_ctr = 0
        for images_train, images_true in self.train_generator:
            # print(images_train[0].shape)
            images_fake = self.generator.predict(images_train)
            valid = np.ones((len(images_train[0]), 1))
            fake = np.zeros((len(images_train[0]), 1))
            log_msg = None
            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_loss_real = self.discriminator.train_on_batch([images_train[0], images_train[1], images_true], valid)
            d_loss_fake = self.discriminator.train_on_batch([images_train[0], images_train[1], images_fake], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            log_mesg = "%d [D loss: %f, acc: %f]" % (step_ctr, d_loss[0], d_loss[1])
            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.adversarial.train_on_batch(images_train, valid)
            
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, g_loss[0], g_loss[1])
            print(log_mesg)
            step_ctr += 1
            if train_steps is not None and step_ctr >= train_steps:
                break

        val_d_losses = np.array([])
        val_g_losses = np.array([])
        step_ctr = 0
        for input_data, images_true in self.val_generator:
            images_fake = self.generator.predict(input_data)
            valid = np.ones((len(input_data[0]), 1))
            fake = np.zeros((len(input_data[0]), 1))

            d_loss_real = self.discriminator.test_on_batch([input_data[0], input_data[1], images_true], valid)
            d_loss_fake = self.discriminator.test_on_batch([input_data[0], input_data[1], images_fake], fake)
            val_d_losses = np.append(val_d_losses, 0.5 * np.add(d_loss_real, d_loss_fake)[1])

            val_g_losses= np.append(val_g_losses, self.adversarial.test_on_batch(input_data, valid)[1])
            step_ctr += 1
            if step_ctr >= val_steps:
                break
        d_loss = np.average(val_d_losses)
        g_loss = np.average(val_g_losses)
        print(f'Epoch {NUM_EPOCHS-save_interval}: val_d_acc: {d_loss}; val_g_acc: {g_loss}')
            
        if save_interval == 0:
             self.DCGAN.save_adversarial_weights("AM.h5")
             self.plot_callback(self.generator)

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    # mnist_dcgan = MNIST_DCGAN(file="AM.h5")
    if len(sys.argv) == 1:
        for i in reversed(range(NUM_EPOCHS)):
            mnist_dcgan.train(train_steps=100, save_interval=i)
    else:
        mnist_dcgan.plot_callback(mnist_dcgan.generator)
        
    # timer.elapsed_time()
    #mnist_dcgan.plot_images(fake=True)
    #mnist_dcgan.plot_images(fake=False, save2file=True)

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
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
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
BATCH_SIZE = 8
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
        self.generator_network = PConvUnet()

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 16
        dropout = 0.4
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling2D((2,2)))
        
        self.D.add(Conv2D(depth*1, 5, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling2D((2,2)))

        self.D.add(Conv2D(depth*2, 5, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling2D((2,2)))

        self.D.add(Conv2D(depth*2, 5, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling2D((2,2)))

        self.D.add(Conv2D(depth*1, 5, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling2D((2,2)))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(32))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        # self.D.summary()
        return self.D
    #Partial convolutional generator
    def build_generator(self):
        if self.G:
            return
            
        self.G = self.generator_network
        self.gen_input = self.G.input
        self.gen_output = self.G.output

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.build_generator()
        self.AM = Model(self.gen_input, self.DM(self.gen_output))
        print(self.AM.summary()) 
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def save_adversarial_weights(self, filepath="AM.h5"):
        self.AM.save(filepath)

    def load_adversarial_weights(self, filepath="AM.h5"):
        self.AM.load(filepath)

# Nothing to do with MNIST
class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
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
            MaskGenerator(512, 512, 1),
            target_size=(512, 512), 
            batch_size=BATCH_SIZE,
            color_mode="grayscale"
        )

        # Create validation generator
        val_datagen = AugmentingDataGenerator(rescale=1./255)
        self.val_generator = val_datagen.flow_from_directory(
            VAL_DIR, 
            MaskGenerator(512, 512, 1), 
            target_size=(512, 512), 
            batch_size=BATCH_SIZE, 
            classes=['val'], 
            seed=42,
            color_mode="grayscale"
        )

        # Create testing generator
        test_datagen = AugmentingDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            TEST_DIR, 
            MaskGenerator(512, 512, 1), 
            target_size=(512, 512), 
            batch_size=BATCH_SIZE, 
            seed=42,
            color_mode="grayscale"
        )
        #yep
        
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

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.G
        
    def plot_callback(self, model):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        model = self.G
        test_data = next(self.test_generator)
        (masked, mask), ori = test_data

        # Show side by side
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,0])
            axes[1].imshow(mask[i,:,:,0] * 1.)
            axes[2].imshow(ori[i,:,:,0])
            plt.show()

        # Get samples & Display them        
        pred_img = model.predict([masked, mask])
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Clear current output and display test images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,0])
            axes[1].imshow(pred_img[i,:,:,0] * 1.)
            axes[2].imshow(ori[i,:,:,0])
            axes[0].set_title('Masked Image')
            axes[1].set_title('Predicted Image')
            axes[2].set_title('Original Image')
                    
            plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
            plt.close()
    
    # One epoch
    # Totally ignore train steps, woopsie
    def train(self, train_steps=2000, batch_size=BATCH_SIZE, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        # Iterate through each batch of training generator
        for inputs, targets in self.train_generator:
            images_train = inputs # [masked, mask]
            print(images_train[0].shape)
            images_fake = self.generator.predict(images_train)
            images_true = targets
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Train the discriminator (real classified as ones and generated as zeros)
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(images_true, valid)
            d_loss_fake = self.discriminator.train_on_batch(images_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            # Reminder, freeze discriminator weights before running generator training, THIS IS MUY IMPORTANTE DO NOT FORGET NEXT TIME
            self.discriminator.trainable = False
            g_loss = self.adversarial.train_on_batch(images_train, valid)
            
            log_mesg = "[D loss: %f, acc: %f]" % (d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, g_loss[0], g_loss[1])
            print(log_mesg)
        if save_interval > 0:
            if save_interval == 0:
                 self.plot_callback(self.generator)
                 self.DCGAN.save_adversarial_weights()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=BATCH_SIZE, save_interval=0)
    # timer.elapsed_time()
    #mnist_dcgan.plot_images(fake=True)
    #mnist_dcgan.plot_images(fake=False, save2file=True)

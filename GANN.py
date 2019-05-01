'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
import gc
import sys
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

from MaskGenerator import MaskGenerator

#SETTINGS

TRAIN_DIR = "TrainingImages/512px/train"
VAL_DIR = "TrainingImages/512px/val"
TEST_DIR = "TrainingImages/512px/test"
BATCH_SIZE = 256
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
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D
    #Partial convolutional generator
    def generator(self):
        if self.G:
            return self.G
        
        return PConvUnet()

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
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

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
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR, 
            MaskGenerator(512, 512, 1),
            target_size=(512, 512), 
            batch_size=BATCH_SIZE,
            color_mode="grayscale"
        )

        # Create validation generator
        val_datagen = AugmentingDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
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
        test_generator = test_datagen.flow_from_directory(
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
        test_data = next(test_generator)
        (masked, mask), ori = test_data

        # # Show side by side
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,0])
            axes[1].imshow(mask[i,:,:,0] * 1.)
            axes[2].imshow(ori[i,:,:,0])
            plt.show()
       # #Not the correct trainign dataset
        # self.x_train = input_data.read_data_sets("mnist",\
        	# one_hot=True).train.images


        #self.DCGAN = DCGAN()
        #self.discriminator =  self.DCGAN.discriminator_model()
        #self.adversarial = self.DCGAN.adversarial_model()
        #self.generator = self.DCGAN.generator()
        
    def plot_callback(model):
        """Called at the end of each epoch, displaying our previous test images,
        as well as their masked predictions and saving them to disk"""
        
        # Get samples & Display them        
        pred_img = model.predict([masked, mask])
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Clear current output and display test images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img[i,:,:,:] * 1.)
            axes[2].imshow(ori[i,:,:,:])
            axes[0].set_title('Masked Image')
            axes[1].set_title('Predicted Image')
            axes[2].set_title('Original Image')
                    
            plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
            plt.close()
    
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    #timer = ElapsedTimer()
    #mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    #timer.elapsed_time()
    #mnist_dcgan.plot_images(fake=True)
    #mnist_dcgan.plot_images(fake=False, save2file=True)

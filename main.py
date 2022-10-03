# Setup the pipeline
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

print(tf.__version__)

# Input pipeline
DATA_FOLDER   = '/home/est_posgrado_manuel.suarez/data/dogs-vs-cats/train'
dog_files = np.array(glob(os.path.join(DATA_FOLDER, 'dog.*.jpg')))
cat_files = np.array(glob(os.path.join(DATA_FOLDER, 'cat.*.jpg')))

BUFFER_SIZE = len(dog_files)
BATCH_SIZE = 20
IMG_WIDTH = 256
IMG_HEIGHT = 256

n_images        = dog_files.shape[0]
steps_per_epoch = n_images//BATCH_SIZE
print('num image files : ', n_images)
print('steps per epoch : ', steps_per_epoch )

def read_and_decode(file):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    # img = img / 255.0
    # img = tf.image.resize(img, INPUT_DIM[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def load_image(file1):
    return read_and_decode(file1)

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image

# Dataset's configuration
# train_dataset = tf.data.Dataset.zip((dog_dataset, cat_dataset))
# train_dataset = train_dataset.shuffle(buffer_size=n_images, reshuffle_each_iteration=True)
# train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.batch(BATCH_SIZE).repeat()

train_dogs = tf.data.Dataset.list_files(dog_files, shuffle=False)
train_dogs = train_dogs.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dogs = train_dogs.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_cats = tf.data.Dataset.list_files(cat_files, shuffle=False)
train_cats = train_cats.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_cats = train_cats.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_dog = next(iter(train_dogs))
sample_cat = next(iter(train_cats))

plt.subplot(121)
plt.title('Dog')
plt.imshow(sample_dog[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Dog with random jitter')
plt.imshow(random_jitter(sample_dog[0]) * 0.5 + 0.5)

print("Plotting dog")
plt.savefig('figure_1.png')

plt.subplot(121)
plt.title('Cat')
plt.imshow(sample_cat[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Cat with random jitter')
plt.imshow(random_jitter(sample_cat[0]) * 0.5 + 0.5)

print("Plotting cat")
plt.savefig('figure_2.png')

# Configure Pix2Pix model
OUTPUT_CHANNELS = 3

# Loss function
# Loss Functions
def discriminator_loss(loss_obj, real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(loss_obj, generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

LAMBDA = 10

class CycleGAN(keras.Model):
    def __init__(self, p_lambda=LAMBDA, summary=False, **kwargs):
        super(CycleGAN, self).__init__(**kwargs)
        self.p_lambda = p_lambda

        # Architecture
        self.generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        self.generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

        self.discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

        # Optimizers
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Loss
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Metric trackers
        self.total_cycle_loss_tracker = tf.keras.metrics.Mean(name="total_cycle_loss")
        self.total_gen_g_loss_tracker = tf.keras.metrics.Mean(name="total_gen_g_loss")
        self.total_gen_f_loss_tracker = tf.keras.metrics.Mean(name="total_gen_f_loss")
        self.disc_x_loss_tracker = tf.keras.metrics.Mean(name="disc_x_loss")
        self.disc_y_loss_tracker = tf.keras.metrics.Mean(name="disc_y_loss")

    @tf.function
    def train_step(self, data):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        real_x, real_y = data
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(self.loss_obj, disc_fake_y)
            gen_f_loss = generator_loss(self.loss_obj, disc_fake_x)

            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(self.loss_obj, disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(self.loss_obj, disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                  self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      self.discriminator_y.trainable_variables))

        # compute progress
        self.total_cycle_loss_tracker.update_state(total_cycle_loss)
        self.total_gen_g_loss_tracker.update_state(total_gen_g_loss)
        self.total_gen_f_loss_tracker.update_state(total_gen_f_loss)
        self.disc_x_loss_tracker.update_state(disc_x_loss)
        self.disc_y_loss_tracker.update_state(disc_y_loss)
        return {
            "total_cycle_loss": self.total_cycle_loss_tracker.result(),
            "total_gen_g_loss": self.total_gen_g_loss_tracker.result(),
            "total_gen_f_loss": self.total_gen_f_loss_tracker.result(),
            "disc_x_loss": self.disc_x_loss_tracker.result(),
            "disc_y_loss": self.disc_y_loss_tracker.result()
        }

cyclegan = CycleGAN(p_lambda=LAMBDA, summary=True)
to_cat = cyclegan.generator_g(sample_dog)
to_dog = cyclegan.generator_f(sample_cat)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_dog, to_cat, sample_cat, to_dog]
title = ['Dog', 'To Cat', 'Cat', 'To Dog']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.savefig('figure_3.png')

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real cat?')
plt.imshow(cyclegan.discriminator_y(sample_cat)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real dog?')
plt.imshow(cyclegan.discriminator_x(sample_dog)[0, ..., -1], cmap='RdBu_r')

plt.savefig('figure_4.png')
print("Model builded")

# Checkpoints
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN
filepath = 'best_weight_model.h5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
terminate = TerminateOnNaN()
callbacks = [checkpoint, terminate]

# Training
EPOCHS = 20

# Train
train_dataset = tf.data.Dataset.zip((train_dogs, train_cats))
cyclegan.compile()
cyclegan.fit(train_dataset,
             batch_size      = BATCH_SIZE,
             epochs          = EPOCHS,
             initial_epoch   = 0,
             steps_per_epoch = steps_per_epoch,
             callbacks       = callbacks)
cyclegan.save_weights("model_vae_faces_1e4.h5")


def generate_images(model, test_input, figname):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(figname)

# Run the trained model on the test dataset
for idx, inp in enumerate(train_dogs.take(5)):
  generate_images(cyclegan.generator_g, inp, f"testimage_{idx+1}")
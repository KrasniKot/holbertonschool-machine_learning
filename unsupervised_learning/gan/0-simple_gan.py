#!/usr/bin/env python3
""" Simple GAN

Requires:
    - tensorflow
    - numpy
    - matplotlib
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """ Defines a Simple GAN """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """ Initializes a Simple Gan
            - generator: generator model
            - discriminator: discriminator model
            - latent_generator: generates a latent vector
                                being the input for the generator
            - real_examples: dataset of real images or data samples
                             that the GAN's trained on
            - batch_size: training batch size
            - disc_iter: iterations the discriminator should perform for each
                         training step the generator does
            learning_rate: learning rate
        """
        super().__init__()  # run the __init__ of keras.Model first
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5     # standard value, but can be changed if necessary
        self.beta_2 = .9     # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        mse = tf.keras.losses.MeanSquaredError
        ons = tf.ones

        self.discriminator.loss = (
            lambda x, y: (
                mse()(x, ons(x.shape)) +
                mse()(y, -1 * ons(y.shape))
            )
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """ Generates a fake sample
            - size: number of fake examples to generate
            - training: determines whether the model is being trained
        """
        if not size:
            size = self.batch_size

        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """ Retrieves a sample of real randomly sorted examples
            - size: sample size
        """
        if not size:
            size = self.batch_size

        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]

        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """ Performs a training step for the Simple GAN """
        for _ in range(self.disc_iter):
            # Real and fake samples
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample()

            # Discriminator cost on real and fake samples
            with tf.GradientTape() as tape:
                # Disrciminator applied to real and fake sample
                real_disc_output = self.discriminator(real_sample)
                fake_disc_output = self.discriminator(fake_sample)

                # Actual discriminator cost
                discr_loss = self.discriminator.loss(x=real_disc_output,
                                                     y=fake_disc_output)

            # Gradient descent is applied once to the discriminator
            disc_gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)

            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

        # Generator cost
        with tf.GradientTape() as tape:
            # Fake sample
            fake_sample = self.get_fake_sample()
            gen_out = self.discriminator(fake_sample, training=True)

            # Generator cost on the sample
            gen_loss = self.generator.loss(gen_out)

        # Gradient Descent is applied to the discriminator
        gen_gradients = tape.gradient(gen_loss,
                                      self.generator.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

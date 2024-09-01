# GANs

## Tasks:

### Simple GAN:
In this task we will define a class ``Simple_GAN`` and test it on some examples. The instruction text is quite long, since we explain the context in details, but the exercise in itself is short: it consists in filling in the ``train_step`` method.

#### The generator and the discriminator networks

We will assume the generator network, the discriminator network, and a generator of latent vectors are already fixed. For example, as the result of the following function:

```
def spheric_generator(nb_points, dim) :
    u=tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u),axis=[1])+10**-8),[nb_points,1])

def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal" ) :
    
    #   Latent generator   
    if latent_type   == "uniform" :
        latent_generator  =  lambda k : tf.random.uniform(shape=(k, gen_shape[0]))
    elif latent_type == "normal" :
        latent_generator  =  lambda k : tf.random.normal(shape=(k, gen_shape[0])) 
    elif latent_type == "spheric" :
        latent_generator  = lambda k : spheric_generator(k,gen_shape[0]) 
    
    #   Generator  
    inputs     = keras.Input(shape=( gen_shape[0] , ))
    hidden     = keras.layers.Dense( gen_shape[1] , activation = 'tanh'    )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden = keras.layers.Dense( gen_shape[i] , activation = 'tanh'    )(hidden)
    outputs    = keras.layers.Dense( gen_shape[-1], activation = 'sigmoid' )(hidden)
    generator  = keras.Model(inputs, outputs, name="generator")
    
    #   Discriminator     
    inputs        = keras.Input(shape=( gen_shape[-1], ))
    hidden        = keras.layers.Dense( gen_shape[-2],   activation = 'tanh' )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden    = keras.layers.Dense( gen_shape[-1*i], activation = 'tanh' )(hidden)
    outputs       = keras.layers.Dense( 1 ,              activation = 'tanh' )(hidden)
    discriminator = keras.Model(inputs, outputs, name="discriminator")
    
    return generator, discriminator, latent_generator
```

The code above produces two networks that are almost symmetric (the first layer of the generator and the last layer of the discriminator differ).

Note that the last layer of the discriminator has the sigmoid for activation function, thus takes values in ``[0, 1]``, while all the others activation functions are the hyperbolic tangent, which takes values in ``[-1, 1]``

For example:
```
generator, discriminator, latent_generator = fully_connected_GenDiscr([1,100,100,2], None)
print(generator.summary())
print(discriminator.summary())
```

produces:

```
Model: "generator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 1)]               0         
_________________________________________________________________
dense (Dense)                (None, 100)               200       
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 202       
=================================================================
Total params: 10,502
Trainable params: 10,502
Non-trainable params: 0
_________________________________________________________________



Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 2)]               0         
_________________________________________________________________
dense_3 (Dense)              (None, 100)               300       
_________________________________________________________________
dense_4 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 101       
=================================================================
Total params: 10,501
Trainable params: 10,501
Non-trainable params: 0
_________________________________________________________________
```

#### The Simple_GAN model

The simple GAN model looks as follows:

```
class Simple_GAN(keras.Model):
    
    def __init__(self, generator , discriminator , latent_generator, real_examples, 
                 batch_size=200, disc_iter=2, learning_rate=.005):
        pass
    
    # generator of real samples of size batch_size
    def get_real_sample(self):
        pass
    
    # generator of fake samples of size batch_size
    def get_fake_sample(self, training=True):
        pass
             
    # overloading train_step()
    def train_step(self,useless_argument): 
        pass
```

The goal of the exercise is to fill in the ``train_step`` method. But before we do so, let us fill in the other methods.

#### The __init__ method, the loss functions and the optimizers

Here is the code for the ``__init__`` method:
```
    def __init( self, generator , discriminator , latent_generator, real_examples, 
                batch_size=200, disc_iter=2, learning_rate=.005):
                
        super().__init__()                         # run the __init__ of Keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate=learning_rate
        self.beta1=.5                               # standard value, but can be changed if necessary
        self.beta2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x , y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
```

The ``super.__init__()`` instruction instanciates some attributes of the model like for example ``self.history``.

The loss of the generator is the mean squared error between ``discriminator(generator(latent_sample)) `` and the (generator) objective value ``1``.

The loss of the discriminator is the mean squared error between ``discriminator(fake_sample)`` and the (discriminator) objective value ``-1``, summed with the mean squared error between ``discriminator(real_sample)`` and the (discriminator) objective value ``1``.

The optimizers are standard Adam optimizers.

#### The get_X_sample methods

A fake sample is just the image of the generator applied to a latent sample:
```
     def get_fake_sample(self, training=False):
        self.generator(self.latent_generator(self.batch_size), training=training)
```

A real sample is a random subset of the set of ``real_examples``:
```
    def get_real_sample(self):
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)
```

#### The train_step method (your shot)

Recall from the lesson on Keras models that to compute a gradient relatively to some variables the scheme is as follows:

```
x = tf.constant([.1, .2, .3, .4])                # x is a tensor 
with tf.GradientTape() as g:
  g.watch( x )                                   # we want to compute the gradient w/r to x
  y = f(x)                                       # y is 1-dimensional, f is a tensorflow function
gradient = g.gradient(y, x)                      # gradient is the gradient of f as at x
```

For example, if we want to train a model ``M`` to minimize a function ``f`` the scheme for one step looks like

```
with tf.GradientTape() as g:
  g.watch( M.trainable_variables )                
  y = f(M)                                        # y is 1-dimensional, f is a tensorflow function
gradient = g.gradient(y, M.trainable_variables)   # get the gradient of f at M
M.optimizer.apply_gradients(zip(gradient, M.trainable_variables))
```

Now one training step of our GANs consists in applying ``discr_iter`` times the gradient descent for the discriminator and then once for the generator.

Thus you are asked to fill in this method:

```
    def train_step(self,useless_argument): 
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the discriminator
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
```

Finally the whole class declaration (with the hole you have to fill in) of the class is:

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class Simple_GAN(keras.Model) :
    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.5                               # standard value, but can be changed if necessary
        self.beta_2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x,y : tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
       
    
    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
             
    # overloading train_step()    
    def train_step(self,useless_argument):
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the discriminator
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
```

### 1. Wasserstein GANs:
In this task we will define a class ``WGAN_clip`` and test it on some examples. Almost all of what we need is already explained in the presentation or in task 0. We just have to modify the losses of the generator and the discriminator, and to clip the weights of the discriminator in ``[-1,1]``.

#### The code to be filled in
Basically we keep the same structure:
```
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class WGAN_clip(keras.Model) :
    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.5                               # standard value, but can be changed if necessary
        self.beta_2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : pass           # <----- new !
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x,y : pass   # <----- new !
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
       
    
    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
        
             
    # overloading train_step()    
    def train_step(self,useless_argument):
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator
            
            # clip the weights (of the discriminator) between -1 and 1    # <----- new !

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the generator
        
        
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
```

#### Your task:
It consists in filling in three holes:

- fill in the ``generator_loss`` function in the ``__init__`` method
- fill in the ``discriminator_loss`` function in the ``__init__`` method
- fill in the ``train_step`` method


#### Reminder:
In the presentation we have seen that for a Wasserstein GAN,

- ``generator_loss(x)`` is the opposite of the mean value of the image x of the discriminator on the image by the generator of a batch of latent vectors
- ``discriminator_loss(x,y)`` is the difference between
 - the mean value of the image y of a batch of real examples by the discriminator and
 - the mean value of the image x by the discriminator of the image by the generator of a batch of latent vectors
- The weights of the discriminator must be clipped in ``[-1,1]``.

#### Hints:
- to compute the losses you should use the function ``tf.math.reduce_mean``
- note that list discriminator.trainable_variables is a list containing the weights and the biases under the form of tensors
- to clip a tensor you should use the function ``tf.clip_by_value``

### 2. Wasserstein GANs with gradient penalty
In this task we will define a class ``WGAN_GP`` and test it on some examples. Almost all of what we need is already explained in the presentation or in tasks 0 and 1. We just have to modify the loss of the discriminator, and forget about clipping.

#### The code to be filled in
So we keep the same structure:
```
class WGAN_GP(keras.Model) :    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005,lambda_gp=10):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.3                              # standard value, but can be changed if necessary
        self.beta_2=.9                              # standard value, but can be changed if necessary
        
        self.lambda_gp        = lambda_gp                                # <---- New !
        self.dims = self.real_examples.shape                             # <---- New !
        self.len_dims=tf.size(self.dims)                                 # <---- New !
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')   # <---- New !
        self.scal_shape=self.dims.as_list()                              # <---- New !
        self.scal_shape[0]=self.batch_size                               # <---- New !
        for i in range(1,self.len_dims):                                 # <---- New !
            self.scal_shape[i]=1                                         # <---- New !
        self.scal_shape=tf.convert_to_tensor(self.scal_shape)            # <---- New !
        
        # define the generator loss and optimizer:
        self.generator.loss  = lambda x : pass                                  # <---- to be filled in                 
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x , y :pass                         # <---- to be filled in  
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
    
    # generator of interpolating samples of size batch_size              # <---- New !
    def get_interpolated_sample(self,real_sample,fake_sample):
        u = tf.random.uniform(self.scal_shape)
        v=tf.ones(self.scal_shape)-u
        return u*real_sample+v*fake_sample
    
    # computing the gradient penalty                                     # <---- New !
    def gradient_penalty(self,interpolated_sample):
        with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_sample)
                pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)      
     
         # overloading train_step()    
    def train_step(self,useless_argument):
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the penalized loss for the discriminator in a tape watching the discriminator's weights
            
                # get a real sample
                # get a fake sample
                # get the interpolated sample (between real and fake computed above)
                                
                # compute the old loss discr_loss of the discriminator on real and fake samples        
                # compute the gradient penalty gp
                # compute the sum new_discr_loss = discr_loss + self.lambda_gp * gp                     
                                
            # apply gradient descent with respect to new_discr_loss once to the discriminator 

        # compute the loss for the generator in a tape watching the generator's weights 
        
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
            
        # apply gradient descent to the discriminator (gp is the gradient penalty)
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp":gp}
```

#### Your task:

It consists in filling in three holes:

- fill in the ``generator_loss`` function in the ``__init__`` method
- fill in the ``discriminator_loss`` function in the ``__init__`` method
- fill in the ``train_step`` method

The two losses are the same as in the WGAN_clip class, and in the training, you just have to add the gradient penalty to the generator loss (inside the tape) as indicated in the commented pseudo code.

### 3. Generating faces:
Write a function`` def convolutional_GenDiscr()`` just like the function ``fully_connected_GenDiscr()`` of task 0 that builds a generator and discriminator as described here under:
- For the generator, the input data will have shape ``(16)``
- For the discriminator, the input data will have shape ``(16,16,1)``
- Use a hyperbolic tangent activation function “tanh” (even in Dense layers)
- In every ``Conv2D``, the ``padding=“same”``
- Returns: the concatenated output of the generator and discriminator

Copy this code and fill in the method ``def convolutional_GenDiscr()``

```
def convolutional_GenDiscr() :


    def generator() :
        #generator model



    def get_discriminator():
        #discriminator model
        

    return get_generator() , get_discriminator()
```

### 4. Our own "This person does not exist": Playing with a pre-trained model
The developers of this project have trained a ``WGAN_GP`` model just as you did in the last task, but for a larger number of epoch ``(150)``, of a larger number of steps ``(200)`` with a small learning rate ``(.0001)`` and stored

- The weights of the generator in a file ``generator.h5``
- The weights of the discriminator in a file ``discriminator.h5``

The aim of this task is to recover the ``WGAN_GP`` model from these weights, and then to play a little bit in the main with this model.

#### Replace the weights
Update your class ``WGAN_GP`` by adding a method ``replace_weight(self,gen_h5,disc_h5)`` that allows you to:

- replace the weights of the generator by the ones that have been stored in the ``.h5`` file ``gen_h5``
- replace the weights of the discriminator by the ones that have been stored in the ``.h5`` file ``gen_h5``

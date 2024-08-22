# Autoencoders

## Tasks:

### 0. "Vanilla" Autoencoder:
Write a function ``def autoencoder(input_dims, hidden_layers, latent_dims):`` that creates an autoencoder:

- ``input_dims`` is an integer containing the dimensions of the model input
- ``hidden_layers`` is a list containing the number of nodes for each hidden layer in the encoder, respectively
- the hidden layers should be reversed for the decoder
- ``latent_dims`` is an integer containing the dimensions of the latent space representation
- Returns: ``encoder``, ``decoder``, ``auto``
  - ``encoder`` is the encoder model
  - ``decoder`` is the decoder model
  - ``auto`` is the full autoencoder model
- The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
- All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid

### 1. Sparse Autoencoder:
Write a function ``def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):`` that creates a sparse autoencoder:

- ``input_dims`` is an integer containing the dimensions of the model input
- ``hidden_layers`` is a list containing the number of nodes for each hidden layer in the encoder, respectively
- the hidden layers should be reversed for the decoder
- ``latent_dims`` is an integer containing the dimensions of the latent space representation
- ``lambtha`` is the regularization parameter used for L1 regularization on the encoded output
- Returns: ``encoder``, ``decoder``, ``auto``
  - ``encoder`` is the encoder model
  - ``decoder`` is the decoder model
  - ``auto`` is the sparse autoencoder model
  - The sparse autoencoder model should be compiled using adam optimization and ``binary cross-entropy`` loss
 - All layers should use a ``relu`` activation except for the last layer in the decoder, which should use ``sigmoid``

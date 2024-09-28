# Attention

## Tasks:

### 0. RNN Encoder:
Create a class ``RNNEncoder`` that inherits from t``ensorflow.keras.layers.Layer`` to encode for machine translation:

Class constructor ``def __init__(self, vocab, embedding, units, batch):``
- ``vocab`` is an integer representing the size of the input vocabulary
- ``embedding`` is an integer representing the dimensionality of the embedding vector
- ``units`` is an integer representing the number of hidden units in the RNN cell
- ``batch`` is an integer representing the batch size
- Sets the following public instance attributes:
   - ``batch`` - the batch size
   - ``units`` - the number of hidden units in the RNN cell
   - ``embedding`` - a ``keras`` Embedding layer that converts words from the vocabulary into an embedding vector
    - ``gru`` - a ``keras`` GRU layer with ``units`` units
     - Should return both the full sequence of outputs as well as the last hidden state
     - Recurrent weights should be initialized with ``glorot_uniform``

Public instance method ``def initialize_hidden_state(self):``
- Initializes the hidden states for the RNN cell to a tensor of zeros
- Returns: a tensor of shape ``(batch, units)`` containing the initialized hidden states

Public instance method ``def call(self, x, initial):``
- ``x`` is a tensor of shape ``(batch, input_seq_len)`` containing the input to the encoder layer as word indices within the vocabulary
- ``initial`` is a tensor of shape ``(batch, units)`` containing the initial hidden state
- Returns: ``outputs``, ``hidden``
   - ``outputs`` is a tensor of shape ``(batch, input_seq_len, units)`` containing the outputs of the encoder
   - ``hidden`` is a tensor of shape ``(batch, units)`` containing the last hidden state of the encoder


### 1. Self Attention:
Create a class ``SelfAttention`` that inherits from ``tensorflow.keras.layers.Layer`` to calculate the attention for machine translation based on this paper:

Class constructor ``def __init__(self, units):``
   - ``units`` is an integer representing the number of hidden units in the alignment model
- Sets the following public instance attributes:
  - ``W`` - a Dense layer with ``units`` units, to be applied to the previous decoder hidden state
  - ``U`` - a Dense layer with ``units`` units, to be applied to the encoder hidden states
  - ``V`` - a Dense layer with ``1`` units, to be applied to the tanh of the sum of the outputs of ``W`` and ``U``

Public instance method def ``call(self, s_prev, hidden_states):``
- ``s_prev`` is a tensor of shape ``(batch, units)`` containing the previous decoder hidden state
- ``hidden_states`` is a tensor of shape ``(batch, input_seq_len, units)`` containing the outputs of  the encoder
- Returns: ``context``, ``weights``
  - ``context`` is a tensor of shape ``(batch, units)`` that contains the context vector for the decoder
  - ``weights`` is a tensor of shape ``(batch, input_seq_len, 1)`` that contains the attention weights

### 2. RNN Decoder:
Create a class ``RNNDecoder`` that inherits from ``tensorflow.keras.layers.Layer`` to decode for machine translation:

Class constructor ``def __init__(self, vocab, embedding, units, batch):``
- ``vocab`` is an integer representing the size of the output vocabulary
- ``embedding`` is an integer representing the dimensionality of the embedding vector
- ``units`` is an integer representing the number of hidden units in the RNN cell
- ``batch`` is an integer representing the batch size
- Sets the following public instance attributes:
   - ``embedding`` - a ``keras`` Embedding layer that converts words from the vocabulary into an embedding vector
   - ``gru`` - a ``keras`` GRU layer with ``units`` units
     - Should return both the full sequence of outputs as well as the last hidden state
     - Recurrent weights should be initialized with ``glorot_uniform``
   - ``F`` - a Dense layer with ``vocab`` units

Public instance method ``def call(self, x, s_prev, hidden_states):``
- ``x`` is a tensor of shape ``(batch, 1)`` containing the previous word in the target sequence as an index of the target vocabulary
- ``s_prev`` is a tensor of shape ``(batch, units)`` containing the previous decoder hidden state
- ``hidden_states`` is a tensor of shape ``(batch, input_seq_len, units)`` containing the outputs of the encoder
- You should use ``SelfAttention = __import__('1-self_attention').SelfAttention``
- You should concatenate the context vector with x in that order
- Returns: ``y``, ``s``
  - ``y`` is a tensor of shape ``(batch, vocab)`` containing the output word as a one hot vector in the target vocabulary
  - ``s`` is a tensor of shape ``(batch, units)`` containing the new decoder hidden state

### 3. Positional Encoding
Write the function ``def positional_encoding(max_seq_len, dm):`` that calculates the positional encoding for a transformer:

- ``max_seq_len`` is an integer representing the maximum sequence length
- ``dm`` is the model depth
- Returns: a ``numpy.ndarray`` of shape ``(max_seq_len, dm)`` containing the positional encoding vectors

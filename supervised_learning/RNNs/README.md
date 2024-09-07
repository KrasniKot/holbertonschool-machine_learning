# Recurrent Neural Networks

## Tasks:

### 0. RNN Cell:
Create the class ``RNNCell`` that represents a cell of a simple RNN:

- class constructor ``def __init__(self, i, h, o):``
  - ``i`` is the dimensionality of the data
  - ``h`` is the dimensionality of the hidden state
  - ``o`` is the dimensionality of the outputs
  - Creates the public instance attributes ``Wh``, ``Wy``, ``bh``, by that represent the weights and biases of the cell
    - ``Wh`` and ``bh`` are for the concatenated hidden state and input data
    - ``Wy`` and by are for the output
  - The weights should be initialized using a random normal distribution in the order listed above
  - The weights will be used on the right side for matrix multiplication
  - The biases should be initialized as zeros
- public instance method ``def forward(self, h_prev, x_t):`` that performs forward propagation for one time step
  - ``x_t`` is a ``numpy.ndarray`` of shape ``(m, i)`` that contains the data input for the cell
    - ``m`` is the batche size for the data
  - ``h_prev`` is a ``numpy.ndarray`` of shape ``(m, h)`` containing the previous hidden state
  - The output of the cell should use a softmax activation function
  - Returns: ``h_next``, ``y``
    - ``h_next`` is the next hidden state
    - ``y`` is the output of the cell
  
### 1. RNN:
Write the function ``def rnn(rnn_cell, X, h_0):`` that performs forward propagation for a simple RNN:

- ``rnn_cell`` is an instance of ``RNNCell`` that will be used for the forward propagation
- ``X`` is the data to be used, given as a ``numpy.ndarray`` of shape ``(t, m, i)``
  - ``t`` is the maximum number of time steps
  - ``m`` is the batch size
  - ``i`` is the dimensionality of the data
- ``h_0`` is the initial hidden state, given as a ``numpy.ndarray`` of shape ``(m, h)``
  - ``h`` is the dimensionality of the hidden state
- Returns: ``H``, ``Y``
  - ``H`` is a ``numpy.ndarray`` containing all of the hidden states
  - ``Y`` is a ``numpy.ndarray`` containing all of the outputs

### 2. GRU Cell
Create the class ``GRUCell`` that represents a gated recurrent unit:

- class constructor ``def __init__(self, i, h, o):``
  - ``i`` is the dimensionality of the data
  - ``h`` is the dimensionality of the hidden state
  - ``o`` is the dimensionality of the outputs
  - Creates the public instance attributes ``Wz``, ``Wr``, ``Wh``, ``Wy``, ``bz``, ``br``, ``bh``, ``by`` that represent the weights and biases of the cell
    - ``Wz`` and ``bz`` are for the update gate
    - ``Wr`` and ``br`` are for the reset gate
    - ``Wh`` and ``bh`` are for the intermediate hidden state
    - ``Wy`` and ``by`` are for the output
  - The weights should be initialized using a random normal distribution in the order listed above
  - The weights will be used on the right side for matrix multiplication
  - The biases should be initialized as zeros
- public instance method ``def forward(self, h_prev, x_t):`` that performs forward propagation for one time step
  - ``x_t`` is a ``numpy.ndarray`` of shape ``(m, i)`` that contains the data input for the cell
    - ``m`` is the batche size for the data
  - ``h_prev`` is a ``numpy.ndarray`` of shape ``(m, h)`` containing the previous hidden state
  - The output of the cell should use a softmax activation function
  - Returns: ``h_next``, ``y``
    - ``h_next`` is the next hidden state
    - ``y`` is the output of the cell
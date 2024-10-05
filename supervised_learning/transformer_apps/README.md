0. Dataset
Create the class Dataset that loads and preps a dataset for machine translation:

Class constructor def __init__(self):
creates the instance attributes:
data_train, which contains the ted_hrlr_translate/pt_to_en tf.data.Dataset train split, loaded as_supervided
data_valid, which contains the ted_hrlr_translate/pt_to_en tf.data.Dataset validate split, loaded as_supervided
tokenizer_pt is the Portuguese tokenizer created from the training set
tokenizer_en is the English tokenizer created from the training set
Create the instance method def tokenize_dataset(self, data): that creates sub-word tokenizers for our dataset:
data is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
pt is the tf.Tensor containing the Portuguese sentence
en is the tf.Tensor containing the corresponding English sentence
Use a pre-trained tokenizer:
use the pretrained model neuralmind/bert-base-portuguese-cased for the portuguese text
use the pretrained model bert-base-uncased for the english text
Train the tokenizers with a maximum vocabulary size of 2**13
Returns: tokenizer_pt, tokenizer_en
tokenizer_pt is the Portuguese tokenizer
tokenizer_en is the English tokenizer

1. Encode Tokens
Update the class Dataset:

Create the instance method def encode(self, pt, en): that encodes a translation into tokens:
pt is the tf.Tensor containing the Portuguese sentence
en is the tf.Tensor containing the corresponding English sentence
The tokenized sentences should include the start and end of sentence tokens
The start token should be indexed as vocab_size
The end token should be indexed as vocab_size + 1
Returns: pt_tokens, en_tokens
pt_tokens is a np.ndarray containing the Portuguese tokens
en_tokens is a np.ndarray. containing the English tokens

2. TF Encode
Update the class Dataset:

Create the instance method def tf_encode(self, pt, en): that acts as a tensorflow wrapper for the encode instance method
Make sure to set the shape of the pt and en return tensors
Update the class constructor def __init__(self):
update the data_train and data_validate attributes by tokenizing the examples

3. Pipeline
Update the class Dataset to set up the data pipeline:

Update the class constructor def __init__(self, batch_size, max_len):
batch_size is the batch size for training/validation
max_len is the maximum number of tokens allowed per example sentence
update the data_train attribute by performing the following actions:
filter out all examples that have either sentence with more than max_len tokens
cache the dataset to increase performance
shuffle the entire dataset using a buffer size equal to 20000.
split the dataset into padded batches of size batch_size
prefetch the dataset using tf.data.experimental.AUTOTUNE to increase performance
update the data_validate attribute by performing the following actions:
filter out all examples that have either sentence with more than max_len tokens
split the dataset into padded batches of size batch_size

4. Create Masks
Create the function def create_masks(inputs, target): that creates all masks for training/validation:

inputs is a tf.Tensor of shape (batch_size, seq_len_in) that contains the input sentence
target is a tf.Tensor of shape (batch_size, seq_len_out) that contains the target sentence
This function should only use tensorflow operations in order to properly function in the training step
Returns: encoder_mask, combined_mask, decoder_mask
encoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) to be applied in the encoder
combined_mask is the tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out) used in the 1st attention block in the decoder to pad and mask future tokens in the input received by the decoder. It takes the maximum between a lookaheadmask and the decoder target padding mask.
decoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) used in the 2nd attention block in the decoder.


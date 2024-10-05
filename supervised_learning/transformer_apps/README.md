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


#!/usr/bin/env python3
""" Load and prepare a dataset for machine translation """

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """ Loads and prepas a dataset for machine translation """

    def __init__(self, batch_size, max_len=20000):
        """ Intitializes the Dataset class
            - batch_size ... batch size for training/validation
            - maximum number of tokens allowed per example sentence
        """
        # Set aliases
        tfenc = self.tf_encode

        # Load training and validation datasets as (input, label)
        dt, dv = tfds.load('ted_hrlr_translate/pt_to_en',
                           split=['train', 'validation'], as_supervised=True)

        # Tokenize training dataset
        tpt, ten = self.tokenize_dataset(dt)

        # ####### Set instance attributes and tokenizers
        # Set tokenizers
        self.tokenizer_pt, self.tokenizer_en = tpt, ten

        # Tokenize training and validation data
        dt = dt.map(tfenc, num_parallel_calls=tf.data.AUTOTUNE)
        dv = dv.map(tfenc, num_parallel_calls=tf.data.AUTOTUNE)

        # Set training and validation datasets
        self.data_train = self._process_exs(dt, batch_size, max_len)
        self.data_valid = self._process_exs(dv, batch_size, max_len, tr=False)
        # #######

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers for our dataset:
            - datat ........... dataset containing tuples of (pt, en) sentences

            >> tokenizer_pt ... trained tokenizer for Portuguese.
            >> tokenizer_en ... trained tokenizer for English.
        """
        # Define aliases
        fp = transformers.AutoTokenizer.from_pretrained

        # Get portugese and english sentences
        ptdata = []
        endata = []
        for pt, en in data.as_numpy_iterator():
            ptdata.append(pt.decode('utf-8'))
            endata.append(en.decode('utf-8'))

        # Load pre-trained tokenizers
        tpt = fp('neuralmind/bert-base-portuguese-cased',
                 use_fast=True, clean_up_tokenization_spaces=True)
        ten = fp('bert-base-uncased', use_fast=True,
                 clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentence iterators
        # and set the corresponding instance attributes
        self.tokenizer_pt = tpt.train_new_from_iterator(ptdata,
                                                        vocab_size=2 ** 13)
        self.tokenizer_en = ten.train_new_from_iterator(endata,
                                                        vocab_size=2 ** 13)

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """ Encondes a translation into tokens
            - pt ..... tf.Tensor containing the Portuguese sentence
            - en ..... tf.Tensor containing the corresponding English sentence

            >> np.ndarray containing the Portuguese tokens
            >> np.ndarray containing the English tokens
        """
        # Decode tf.Tensor to strings
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')

        # Tokenize sentences with no special tokens
        ptkns = self.tokenizer_pt.encode(pt, add_special_tokens=False)
        etkns = self.tokenizer_en.encode(en, add_special_tokens=False)

        # ####### Insert sentence start and end tokens
        # Extract vocab size
        ptvsize = self.tokenizer_pt.vocab_size
        envsize = self.tokenizer_en.vocab_size

        # Inserting the SOS (vocab size) and EOS (vocab size + 1) tokens
        ptkns = [ptvsize] + ptkns + [ptvsize + 1]
        etkns = [envsize] + etkns + [envsize + 1]
        # #######

        return ptkns, etkns

    def tf_encode(self, pt, en):
        """ Acts as a tensorflow wrapper for the encode instance method
            - pt ..... tf.Tensor containing the Portuguese sentence
            - en ..... tf.Tensor containing the corresponding English sentence

            >> Tokenized Portuguese sentence tensor
            >> Tokenized English sentence tensor
        """
        # Wrap the self.encode function
        ptkns, enkns = tf.py_function(func=self.encode,
                                      inp=[pt, en], Tout=[tf.int64, tf.int64])

        # Set the shape of the tensors to [None], a 1D array of variable length
        ptkns.set_shape([None])
        enkns.set_shape([None])

        return ptkns, enkns

    def _process_exs(self, data, bs, mx=20000, tr=True):
        """ Processes the examples
            - data ...... examples
            - bs ........ batch size
            - mx ... maximum accepted length for an exmample
            - tr ........ determines if dealing with training data

            > Processed examples
        """
        # Set aliases
        bm = tf.boolean_mask
        rs = tf.reduce_sum
        ct = tf.cast
        la = tf.logical_and

        # Filter out those examples longer than the specified max_len
        data = data.filter(lambda x, y: la(tf.size(x) <= mx, tf.size(y) <= mx))

        # Cache and shuffle the data to increase performance
        if tr:
            data = data.cache().shuffle(buffer_size=20000)

        # Split the data into padded batches of size batch_size
        data = data.padded_batch(batch_size=bs, padded_shapes=([None], [None]))

        # Prefetch the data to increase performance if train
        return data.prefetch(tf.data.AUTOTUNE) if tr else data

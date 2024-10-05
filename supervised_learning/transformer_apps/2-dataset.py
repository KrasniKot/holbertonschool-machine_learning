#!/usr/bin/env python3
""" Load and prepare a dataset for machine translation """

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """ Loads and prepas a dataset for machine translation """

    def __init__(self):
        """ Intitializes the Dataset class """
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

        # Set tokenized training data
        self.data_train = dt.map(tfenc, num_parallel_calls=tf.data.AUTOTUNE)

        # Set tokenized validation data
        self.data_valid = dv.map(tfenc, num_parallel_calls=tf.data.AUTOTUNE)
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
        tpt = fp('neuralmind/bert-base-portuguese-cased', use_fast=True,
                 clean_up_tokenization_spaces=True)
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
        ptkns, enkns = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64])

        # Set the shape of the tensors to [None], a 1D array of variable length
        ptkns.set_shape([None])
        enkns.set_shape([None])

        return ptkns, enkns

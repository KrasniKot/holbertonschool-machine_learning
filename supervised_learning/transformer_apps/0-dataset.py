#!/usr/bin/env python3
""" Load and prepare a dataset for machine translation """

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """ Loads and prepas a dataset for machine translation """

    def __init__(self):
        """ Intitializes the Dataset class """
        # ####### Prepare datasets
        # Load training and validation datasets as (input, label)
        dt, dv = tfds.load('ted_hrlr_translate/pt_to_en',
                           split=['train', 'validation'], as_supervised=True)
        # Tokenize training dataset
        tpt, ten = self.tokenize_dataset(dt)

        # Set instance attributes and tokenizers
        self.data_train, self.data_valid = dt, dv
        self.tokenizer_pt, self.tokenizer_en = tpt, ten
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

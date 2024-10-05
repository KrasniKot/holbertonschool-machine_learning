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
        ten, tpt = self.tokenize_dataset(dt)

        # Set instance attributes and tokenizers
        self.data_train, self.data_valid = dt, dv
        self.tokenizer_en, self.tokenizer_pt = ten, tpt
        # #######

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers for our dataset:
            - datat ........... dataset containing tuples of (pt, en) sentences

            >> tokenizer_pt ... trained tokenizer for Portuguese.
            >> tokenizer_en ... trained tokenizer for English.
        """
        bfc = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus

        # ####### Build a subword text tokenizer of vocab size 2 **12 (32768),
        # built from the corpus of sentences in the datasets
        tpt = bfc((pt.numpy() for pt, _ in data), target_vocab_size=2 ** 15)
        ten = bfc((en.numpy() for _, en in data), target_vocab_size=2 ** 15)

        # set the instance tokenizers
        self.tokenizer_pt = tpt
        self.tokenizer_en = ten
        # #######

        return self.tokenizer_pt, self.tokenizer_en

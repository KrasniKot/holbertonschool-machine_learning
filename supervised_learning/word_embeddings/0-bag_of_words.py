#!/usr/bin/env python3
""" Function that creates a bag of words """

import numpy as np
import re


def normalize_word(word):
    """ Normalizes words """
    # Convert to lowercase
    word = word.lower()

    # Remove possessive form (e.g., children's -> children)
    word = re.sub(r"'s$", "", word)

    # Remove punctuation (e.g., awesome! -> awesome)
    word = re.sub(r"[^\w\s]", "", word)

    return word


def tokenize(sentence):
    """ Tokenizes and normalizes the sentences """
    return [normalize_word(word) for word in sentence.split()]


def bag_of_words(sentences, vocab=None):
    """ Creates a Bag Of Words embedding matrix.
        Returns the matrix and the features used for the embedding.
        - sentences: list of sentences to analyze
        - vocab: list of the vocabulary words to use for the analysis
    """
    # Tokenize sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    # If vocab is not provided, create it from all unique words in sentences
    if vocab is None:
        vocab = sorted(set(w for s in tokenized_sentences for w in s))

    # Create a mapping from word to index for the vocab
    word_index = {word: i for i, word in enumerate(vocab)}

    # Initialize the embeddings matrix with shape (s, f)
    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    # Populate the embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_index:
                # Add 1 for each time a word appears in a sentenece
                embeddings[i][word_index[word]] += 1

    return embeddings, np.array(vocab)

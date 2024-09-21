#!/usr/bin/env python3
""" Function that creates a TDIFD embedding """

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Creates a TD-IDF embedding and returns the embeddings and features
        - sentences: list of sentences to analyze
        - vocab: list of the vocabulary words to use for the analysis
    """
    # Initialize the TfidfVectorizer with the optional vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit the vectorizer to the sentences
    # and transform them into TF-IDF features
    X = vectorizer.fit_transform(sentences)

    # Convert the sparse matrix to a dense array (if needed)
    # and get the feature names
    return X.toarray(), vectorizer.get_feature_names_out()

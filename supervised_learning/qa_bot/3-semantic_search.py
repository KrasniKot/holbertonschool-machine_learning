#!/usr/bin/env python3
""" Performs semantic search on a corpus of documents """

import os
import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_corpus(cp):
    """ Load documents from a specified corpus path
        - cp ...... corpus path
    """
    documents = []
    for gfile in os.listdir(cp):
        if gfile.endswith('.md'):
            with open(os.path.join(cp, gfile), 'r', encoding='utf-8') as file:
                documents.append(file.read())

    return documents


def semantic_search(corpus_path, sentence):
    """ Perform semantic search to find the most
        similar document to the input sentence
        - corpus_path ...... path to the corpus of reference documents
                             on which to perform semantic search
        - sentence ......... sentence from which to perform semantic search
    """
    # Load the documents
    documents = load_corpus(corpus_path)

    # Load pretrained model
    model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    # Embed the corpus documents and the input sentence
    document_embs = model.encode(documents, convert_to_tensor=True)
    sentence_emb = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities between the sentence and document
    similarities = util.pytorch_cos_sim(sentence_emb, document_embs)

    # Find the index of the most similar document
    most_similar_index = similarities.argmax()

    # Return the text of the most similar document
    return documents[most_similar_index]

#!/usr/bin/env python3
""" Answers questions from multiple reference texts """

import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util


def get_answer(question, reference):
    """ Finds a snippet of text within a reference document to answer questions
        - question ..... string containing the question to answer
        - reference .... string containing the reference document
                         from which to find the answer
        > a string containing the answer, or None if no asnwer found
    """
    # Set aliases
    BT = BertTokenizer.from_pretrained

    # ####### Load necessary models
    # Load the BERT model which is pretrained on question answering
    bert_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load the pre-trained tokenizer from Hugging Face
    tokenizer = BT('bert-large-uncased-whole-word-masking-finetuned-squad')
    # #######

    # Tokenize the input question and reference (context)
    inputs = tokenizer(question,
                       reference,
                       return_tensors='tf',
                       truncation=True,
                       padding=True)

    # Perform inference on the model to get start and end logits
    outputs = bert_model([inputs["input_ids"],
                          inputs["attention_mask"],
                          inputs["token_type_ids"]])

    # Extract the start and end logits
    start_logits, end_logits = outputs[0: 2]

    # Get the input sequence length
    seqlen = inputs["input_ids"].shape[1]

    # Convert logits to indices of start and end tokens
    start_index = tf.math.argmax(start_logits[0, 1:seqlen - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:seqlen - 1]) + 1

    # Get the answer tokens
    tkns = inputs["input_ids"][0][start_index: end_index + 1]

    # Convert token indices back into words
    answer = tokenizer.decode(tkns,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # If the model fails to find an answer, return None
    if not answer.strip():
        return None

    return answer


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
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # Embed the corpus documents and the input sentence
    document_embs = model.encode(documents, convert_to_tensor=True)
    sentence_emb = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities between the sentence and document
    similarities = util.pytorch_cos_sim(sentence_emb, document_embs)

    # Find the index of the most similar document
    most_similar_index = similarities.argmax()

    # Return the text of the most similar document
    return documents[most_similar_index]


def question_answer(corpus_path):
    """  Answers questions from multiple reference texts
        - corpus_path .... path to the corpus of reference documents
    """

    LTAKING = ["exit", "quit", "goodbye", "bye"]  # Leave-taking expressions

    leaving = False
    while not leaving:
        question = input("Q: ").lower()

        if question in LTAKING:
            leaving = True
            response = "Goodbye"

        else:
            reference = semantic_search(corpus_path, question)
            response = get_answer(question=question, reference=reference)
            idk = "Sorry, I do not understand your question."
            response = response if response else idk

        print(f"A: {response}")

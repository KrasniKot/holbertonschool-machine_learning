#!/usr/bin/env python3
""" Function that calculates the n-gram BLUE score """

import math
from collections import Counter


def get_ngrams(words, n):
    """ Generates the n-grams for a list of words """
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def calculate_precision(sentence_ngrams, references, n):
    """ Calculates the precision
        - sentence_ngrams: List of n-grams generated from the sentence
        - references: List of reference translations
        - n: Size of the n-grams
    """
    sentence_counts = Counter(sentence_ngrams)
    max_counts = Counter()

    # Calculate maximum n-gram counts from reference translations
    for reference in references:
        reference_ngrams = get_ngrams(reference, n)
        ref_counts = Counter(reference_ngrams)
        for ngram in sentence_counts:
            max_counts[ngram] = max(max_counts[ngram], ref_counts[ngram])

    # Clipped count: the minimum between n-grams in the sentence
    #                and their max occurrence in any reference
    clipped_count = sum(min(
        sentence_counts[ngrm], max_counts[ngrm]) for ngrm in sentence_counts)

    # Total n-grams in the sentence
    total_ngrams = len(sentence_ngrams)

    # Avoid division by zero if no n-grams are present
    if total_ngrams == 0:
        return 0

    return clipped_count / total_ngrams


def calculate_brevity_penalty(references, sentence_length):
    """ Calculates the bravity penalty
        - references: List of references
        - sentence_length: Proposed model sentence legth
    """
    ref_lengths = [len(ref) for ref in references]

    # Find the closest reference length
    closest_ref_length = min(
        ref_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len)
        )

    if sentence_length > closest_ref_length:
        return 1
    else:
        return math.exp(1 - closest_ref_length / sentence_length)


def ngram_bleu(references, sentence, n):
    """ Calculates the n-gram BlEU score
        - reference: list of reference translations
        - sentence: list containing the model proposed sentence
        - n: size of the n-gram to use for evaluation
    """
    # Step 1: Generate n-grams for the sentence
    sentence_ngrams = get_ngrams(sentence, n)

    # Step 2: Calculate precision for the n-grams
    precision = calculate_precision(sentence_ngrams, references, n)

    # Step 3: Calculate the brevity penalty
    brevity_penalty = calculate_brevity_penalty(references, len(sentence))

    # Step 4: Compute the final BLEU score as precision * brevity penalty
    return brevity_penalty * precision

#!/usr/bin/env python3
""" Function that calculates the Unigram BLEU score """

from collections import Counter
import math


def uni_bleu(references, sentence):
    """ Calculates the Unigram BLEU score
        - reference: list of reference translations
        - sentence: list containing the model proposed sentence
    """
    # Step 1: Count unigrams in the sentence
    sentence_counter = Counter(sentence)

    # Step 2: Get the maximum possible count for each word in the references
    max_counter = Counter()
    for ref in references:
        ref_counter = Counter(ref)
        for word in sentence_counter:
            max_counter[word] = max(max_counter[word], ref_counter[word])

    # Step 3: Count the total number of unigrams that are matched
    clipped_count = 0
    for word in sentence_counter:
        clipped_count += min(sentence_counter[word], max_counter[word])

    # Precision calculation
    precision = clipped_count / len(sentence)

    # Step 4: Calculate the brevity penalty
    ref_lengths = [len(ref) for ref in references]
    sent_length = len(sentence)

    # Find the closest reference length to the model proposed sentence length
    closest_ref_length = min(
        ref_lengths,
        key=lambda ref_len: (abs(ref_len - sent_length), ref_len))

    # Calculate bravity penalty to discourage overly short translations.
    if sent_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / sent_length)

    # Step 5: Calculate the final unigram BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score

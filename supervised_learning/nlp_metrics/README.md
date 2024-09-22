# Natural Language Processing - Evaluation Metrics

## Tasks:

### 0. Unigram BLEU score:
Write the function ``def uni_bleu(references, sentence):`` that calculates the unigram BLEU score for a sentence:

- ``references`` is a list of reference translations
  - each reference translation is a list of the words in the translation
- ``sentence`` is a list containing the model proposed sentence
- Returns: the unigram BLEU score

### 1. N-gram BLEU score
Write the function ``def ngram_bleu(references, sentence, n):`` that calculates the n-gram BLEU score for a sentence:

- ``references`` is a list of reference translations
  - each reference translation is a list of the words in the translation
- ``sentence`` is a list containing the model proposed sentence
- ``n`` is the size of the n-gram to use for evaluation
- Returns: the n-gram BLEU score

### 
# Natural Language Processing - Word Embeddings

## Tasks:

### 0. Bag Of Words:
Write a function ``def bag_of_words(sentences, vocab=None):`` that creates a bag of words embedding matrix:

- ``sentences`` is a list of sentences to analyze
- ``vocab`` is a list of the vocabulary words to use for the analysis
  - If ``None``, all words within sentences should be used
- Returns: ``embeddings``, ``features``
  - ``embeddings`` is a ``numpy.ndarray`` of shape ``(s, f)`` containing the embeddings
    - ``s`` is the number of sentences in ``sentences``
    - ``f`` is the number of features analyzed
  - features is a list of the features used for ``embeddings``
- You are not allowed to use ``genism`` library.


### 1. TF-IDF:
Write a function ``def tf_idf(sentences, vocab=None):`` that creates a TF-IDF embedding:

- ``sentences`` is a list of sentences to analyze
- ``vocab`` is a list of the vocabulary words to use for the analysis
  - If ``None``, all words within sentences should be used
- Returns: ``embeddings``, ``features``
  - embeddings is a ``numpy.ndarray`` of shape ``(s, f)`` containing the embeddings
    - ``s`` is the number of sentences in ``sentences``
    - ``f`` is the number of features analyzed
  - ``features`` is a list of the features used for ``embeddings``

### 2. Train Word2Vec:


### 3. Extract Word2Vec:


### 4. FastText:


### 5. ELMo:


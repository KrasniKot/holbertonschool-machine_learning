# QA BOT

## Tasks

### 0. Question Answering:
Write a function ``def question_answer(question, reference):`` that finds a snippet of text within a reference document to answer a question:

- ``question`` is a string containing the question to answer
- ``reference`` is a string containing the reference document from which to find the answer
- Returns: a string containing the answer
- If no answer is found, return ``None``
- Your function should use the ``bert-uncased-tf2-qa`` model from the ``tensorflow-hub`` library
- Your function should use the pre-trained ``BertTokenizer``, ``bert-large-uncased-whole-word-masking-finetuned-squad`` from the ``transformers`` library

### 1. Create the loop:
Create a script that takes in input from the user with the prompt ``Q:`` and prints ``A:`` as a response. If the user inputs ``exit``, ``quit``, ``goodbye``, or ``bye``, case insensitive, print ``A: Goodbye`` and exit.

### 2. Answer Questions:
### 3. Semantic Search:
### 4. Multi-reference Question Answering:
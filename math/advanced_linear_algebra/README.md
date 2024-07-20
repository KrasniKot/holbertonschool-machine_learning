# Advanced Linear Algebra

## Tasks

### 0. Determinant:
Write a function ``def determinant(matrix):`` that calculates the determinant of a ``matrix``:

- ``matrix`` is a list of lists whose determinant should be calculated
- If ``matrix`` is not a list of lists, raise a ``TypeError`` with the message ``matrix must be a list of lists``
- If ``matrix`` is not square, raise a ``ValueError`` with the message ``matrix must be a square matrix``
- The list ``[[]]`` represents a 0x0 ``matrix``
- Returns: the determinant of ``matrix``

### 1. Minor:
Write a function ``def minor(matrix):`` that calculates the minor matrix of a ``matrix``:

- ``matrix`` is a list of lists whose minor ``matrix`` should be calculated
- If ``matrix`` is not a list of lists, raise a ``TypeError`` with the message ``matrix must be a list of lists``
- If ``matrix`` is not square or is empty, raise a ``ValueError`` with the ``message matrix must be a non-empty square matrix``
- Returns: the minor matrix of ``matrix```

### 2. Cofactor:
Write a function ``def cofactor(matrix):`` that calculates the cofactor matrix of a matrix:

- ``matrix`` is a list of lists whose cofactor ``matrix`` should be calculated
- If ``matrix`` is not a list of lists, raise a ``TypeError`` with the message ``matrix must be a list of lists``
- If ``matrix`` is not square or is empty, raise a ``ValueError`` with the message ``matrix must be a non-empty square matrix``
- Returns: the cofactor matrix of ``matrix``

## 3. Adjugate:
Write a function ``def adjugate(matrix):`` that calculates the adjugate matrix of a matrix:

- ``matrix`` is a list of lists whose adjugate matrix should be calculated
- If ``matrix`` is not a list of lists, raise a ``TypeError`` with the message ``matrix must be a list of lists``
- If ``matrix`` is not square or is empty, raise a ``ValueError`` with the message ``matrix must be a non-empty square matrix``
- Returns: the adjugate matrix of ``matrix``

## 4. Inverse:
Write a function ``def inverse(matrix):`` that calculates the inverse of a matrix:

- ``matrix`` is a list of lists whose inverse should be calculated
- If ``matrix`` is not a list of lists, raise a ``TypeError`` with the message ``matrix must be a list of lists``
- If ``matrix`` is not square or is empty, raise a ``ValueError`` with the message ``matrix must be a non-em`pty square matrix``
- Returns: the inverse of ``matrix``, or ``None`` if ``matrix`` is singular

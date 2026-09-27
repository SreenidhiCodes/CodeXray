"""
text_similarity.py
------------------
Text-based Python code similarity for CodeXray.

This module provides:
    - Normalized text similarity
    - Line similarity
    - Token similarity
    - Jaccard similarity
    - Overall text similarity

The main function used by CodeXray is:

    text_similarity(code1, code2)

It returns a similarity score between 0 and 1.
"""

import difflib
import keyword
import re


# Python keywords are kept because they are useful for
# comparing the structure of two programs.
PYTHON_KEYWORDS = set(keyword.kwlist)


def remove_comments(code):
    """
    Remove Python comments from source code.

    This is a simple text-based operation and is intended
    for similarity comparison rather than code execution.
    """

    code = str(code)

    # Remove comments that begin with #.
    code = re.sub(r"#.*", "", code)

    return code


def normalize_text(code):
    """
    Normalize Python source code for text comparison.

    Operations:
        - Remove comments
        - Convert whitespace to single spaces
        - Convert text to lowercase
        - Remove leading/trailing whitespace

    Returns:
        str: normalized source code
    """

    code = remove_comments(code)

    # Replace all whitespace with one space.
    code = re.sub(r"\s+", " ", code)

    # Remove leading/trailing spaces.
    code = code.strip()

    # Case-insensitive comparison.
    return code.lower()


def tokenize_code(code):
    """
    Convert Python code into simple lexical tokens.

    The tokenizer keeps:
        - identifiers
        - numbers
        - operators
        - punctuation
        - Python keywords

    Returns:
        list: tokens
    """

    code = remove_comments(code)

    # Convert source into useful lexical pieces.
    tokens = re.findall(
        r"""
        [A-Za-z_][A-Za-z0-9_]*     # identifier or keyword
        |
        \d+(?:\.\d+)?              # number
        |
        ==|!=|<=|>=|//|\*\*|->     # multi-character operators
        |
        [-+*/%=<>()[\]{},.:;]       # single-character operators
        """,
        code,
        re.VERBOSE,
    )

    return tokens


def normalize_tokens(tokens):
    """
    Normalize tokens for structural comparison.

    Ordinary variable/function identifiers are replaced with
    IDENTIFIER, while Python keywords remain unchanged.

    Example:

        def add(a, b):
            return a + b

    becomes approximately:

        def IDENTIFIER ( IDENTIFIER , IDENTIFIER ) :
        return IDENTIFIER + IDENTIFIER

    This helps variable-renamed versions receive a higher
    similarity score.
    """

    normalized = []

    for token in tokens:

        # Keep Python keywords.
        if token in PYTHON_KEYWORDS:
            normalized.append(token)

        # Keep numbers.
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            normalized.append("NUMBER")

        # Keep operators and punctuation.
        elif re.fullmatch(
            r"==|!=|<=|>=|//|\*\*|->|[-+*/%=<>()[\]{},.:;]",
            token,
        ):
            normalized.append(token)

        # Replace ordinary identifiers.
        else:
            normalized.append("IDENTIFIER")

    return normalized


def sequence_similarity(code1, code2):
    """
    Compare normalized source text using difflib.

    Returns:
        float: score between 0 and 1
    """

    text1 = normalize_text(code1)
    text2 = normalize_text(code2)

    if not text1 and not text2:
        return 1.0

    if not text1 or not text2:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def line_similarity(code1, code2):
    """
    Compare code line-by-line.

    Whitespace differences are ignored.

    Returns:
        float: score between 0 and 1
    """

    lines1 = [
        normalize_text(line)
        for line in str(code1).splitlines()
        if normalize_text(line)
    ]

    lines2 = [
        normalize_text(line)
        for line in str(code2).splitlines()
        if normalize_text(line)
    ]

    if not lines1 and not lines2:
        return 1.0

    if not lines1 or not lines2:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        lines1,
        lines2
    ).ratio()


def token_similarity(code1, code2):
    """
    Compare normalized token sequences.

    Identifiers are normalized so variable renaming does not
    completely destroy the similarity score.

    Returns:
        float: score between 0 and 1
    """

    tokens1 = normalize_tokens(
        tokenize_code(code1)
    )

    tokens2 = normalize_tokens(
        tokenize_code(code2)
    )

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        tokens1,
        tokens2
    ).ratio()


def jaccard_similarity(code1, code2):
    """
    Calculate Jaccard similarity using normalized tokens.

    Formula:

        intersection / union

    Returns:
        float: score between 0 and 1
    """

    tokens1 = set(
        normalize_tokens(
            tokenize_code(code1)
        )
    )

    tokens2 = set(
        normalize_tokens(
            tokenize_code(code2)
        )
    )

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)


def text_similarity(code1, code2):
    """
    Calculate the overall text-based similarity.

    Four text-based measurements are combined:

        1. Sequence similarity
        2. Line similarity
        3. Token similarity
        4. Jaccard similarity

    Each component contributes equally.

    Returns:
        float: overall score between 0 and 1
    """

    sequence_score = sequence_similarity(
        code1,
        code2
    )

    line_score = line_similarity(
        code1,
        code2
    )

    token_score = token_similarity(
        code1,
        code2
    )

    jaccard_score = jaccard_similarity(
        code1,
        code2
    )

    overall_score = (
        sequence_score
        + line_score
        + token_score
        + jaccard_score
    ) / 4

    return overall_score


def compare_code(code1, code2):
    """
    Return all text similarity measurements.

    Useful for testing and evaluation.
    """

    return {
        "sequence_similarity": sequence_similarity(
            code1,
            code2
        ),
        "line_similarity": line_similarity(
            code1,
            code2
        ),
        "token_similarity": token_similarity(
            code1,
            code2
        ),
        "jaccard_similarity": jaccard_similarity(
            code1,
            code2
        ),
        "text_similarity": text_similarity(
            code1,
            code2
        ),
    }


if __name__ == "__main__":

    # Example 1: similar code with renamed variables.
    code_a = """
def add(a, b):
    return a + b
"""

    code_b = """
def add(x, y):
    return x + y
"""

    # Example 2: structurally different code.
    code_c = """
def multiply(numbers):
    result = 0

    for number in numbers:
        result = result + number

    return result
"""

    print("=" * 50)
    print("CodeXray Text Similarity Test")
    print("=" * 50)

    print("\nComparing similar code:")
    print(compare_code(code_a, code_b))

    print("\nComparing different code:")
    print(compare_code(code_a, code_c))
"""
preprocessor.py
---------------
Prepares Python source code for CodeXray analysis.

Functions:
    validate_python(code) -> dict
    clean_code(code) -> str
    tokenize_code(code) -> list
    preprocess_code(code) -> dict
"""

import ast
import io
import tokenize


def _decode_code(code):
    """
    Accept Python source as either str or bytes.

    If bytes are provided, try to detect the Python source encoding.
    """
    if isinstance(code, str):
        return code

    if isinstance(code, bytes):
        try:
            encoding, _ = tokenize.detect_encoding(
                io.BytesIO(code).readline
            )
            return code.decode(encoding)
        except (SyntaxError, UnicodeDecodeError):
            return code.decode("utf-8", errors="replace")

    raise TypeError("code must be a str or bytes object")


def validate_python(code):
    """
    Check whether the supplied Python code is syntactically valid.

    Returns:
        {
            "valid": True/False,
            "error": None or error message
        }
    """
    try:
        source = _decode_code(code)
        ast.parse(source)

        return {
            "valid": True,
            "error": None
        }

    except (SyntaxError, UnicodeDecodeError, TypeError) as error:
        return {
            "valid": False,
            "error": str(error)
        }


def clean_code(code):
    """
    Remove comments and unnecessary blank lines while preserving
    Python string contents.
    """
    source = _decode_code(code)

    try:
        token_stream = tokenize.generate_tokens(
            io.StringIO(source).readline
        )

        cleaned_tokens = []

        for token in token_stream:
            # Remove comments.
            if token.type == tokenize.COMMENT:
                continue

            # Remove blank-line tokens.
            if token.type == tokenize.NL:
                continue

            cleaned_tokens.append(token)

        cleaned = tokenize.untokenize(cleaned_tokens)

        # Remove completely empty lines and stray backslash lines.
        lines = []

        for line in cleaned.splitlines():
            line = line.rstrip()

            if not line.strip():
                continue

            if line.strip() == "\\":
                continue

            lines.append(line)

        return "\n".join(lines)

    except (tokenize.TokenError, IndentationError):
        # If tokenization fails, return a simple cleaned version.
        lines = []

        for line in source.splitlines():
            if line.strip():
                lines.append(line.rstrip())

        return "\n".join(lines)


def tokenize_code(code):
    """
    Convert Python source code into a list of meaningful tokens.

    Comments, blank-line tokens, indentation markers, and newline
    markers are excluded because they are not useful for basic
    token-level similarity.
    """
    source = _decode_code(code)

    tokens = []

    try:
        token_stream = tokenize.generate_tokens(
            io.StringIO(source).readline
        )

        ignored_token_types = {
            tokenize.ENCODING,
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        }

        for token in token_stream:
            if token.type not in ignored_token_types:

                # Ignore empty token strings.
                if token.string.strip():
                    tokens.append(token.string)

    except (tokenize.TokenError, IndentationError):
        return []

    return tokens


def preprocess_code(code):
    """
    Run the complete preprocessing pipeline.

    Returns:
        {
            "valid": bool,
            "error": str or None,
            "clean_code": str,
            "tokens": list
        }
    """
    validation = validate_python(code)

    cleaned = clean_code(code)
    tokens = tokenize_code(cleaned)

    return {
        "valid": validation["valid"],
        "error": validation["error"],
        "clean_code": cleaned,
        "tokens": tokens,
    }


if __name__ == "__main__":
    sample_code = '''
# This is a comment

x = 10
y = 20  # Another comment

print(x + y)
'''

    result = preprocess_code(sample_code)

    print("=" * 50)
    print("CodeXray Preprocessor Test")
    print("=" * 50)

    print("\nValid Python:", result["valid"])
    print("Error:", result["error"])

    print("\nCleaned Code:")
    print(result["clean_code"])

    print("\nTokens:")
    print(result["tokens"])

    print("\nNumber of tokens:", len(result["tokens"]))
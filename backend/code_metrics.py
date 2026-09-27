"""
code_metrics.py
---------------
Calculates basic Python code metrics for CodeXray.

Metrics:
    - Lines of Code
    - Number of Functions
    - Number of Loops
    - Number of Conditions
    - Number of Variables
    - Number of Imports
    - Number of Classes
    - Cyclomatic Complexity

The module uses Python's built-in AST module.
"""

import ast


def calculate_metrics(code):
    """
    Calculate metrics for a Python source-code string.

    Returns:
        dict containing all calculated metrics.
    """

    # Parse the Python code into an Abstract Syntax Tree.
    tree = ast.parse(code)

    # ---------------------------------------------------------
    # 1. Lines of Code
    # ---------------------------------------------------------
    lines_of_code = 0

    for line in code.splitlines():
        # Count non-empty lines.
        if line.strip():
            lines_of_code += 1

    # ---------------------------------------------------------
    # 2. Initialize counters
    # ---------------------------------------------------------
    functions = 0
    loops = 0
    conditions = 0
    variables = 0
    imports = 0
    classes = 0

    # Cyclomatic complexity starts at 1.
    complexity = 1

    # ---------------------------------------------------------
    # 3. Walk through the AST
    # ---------------------------------------------------------
    for node in ast.walk(tree):

        # Functions
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            functions += 1

        # Loops
        elif isinstance(
            node,
            (ast.For, ast.While, ast.AsyncFor)
        ):
            loops += 1
            complexity += 1

        # Conditions
        elif isinstance(node, ast.If):
            conditions += 1
            complexity += 1

        # Conditional expression:
        # example: x if condition else y
        elif isinstance(node, ast.IfExp):
            conditions += 1
            complexity += 1

        # Imports
        elif isinstance(
            node,
            (ast.Import, ast.ImportFrom)
        ):
            imports += 1

        # Classes
        elif isinstance(node, ast.ClassDef):
            classes += 1

        # Variables
        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
            )
        ):
            variables += count_assignment_targets(node)

        # Boolean conditions such as:
        # if x > 5 and y < 10:
        #
        # Each additional boolean condition increases
        # cyclomatic complexity.
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)

        # Exception handlers:
        # try:
        #     ...
        # except:
        #     ...
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1

    return {
        "lines_of_code": lines_of_code,
        "functions": functions,
        "loops": loops,
        "conditions": conditions,
        "variables": variables,
        "imports": imports,
        "classes": classes,
        "cyclomatic_complexity": complexity,
    }


def count_assignment_targets(node):
    """
    Count variables introduced by an assignment.

    Examples:

        x = 10
            -> 1 variable

        x = y = 10
            -> 2 variables

        a, b = values
            -> 2 variables
    """

    if isinstance(node, ast.Assign):
        total = 0

        for target in node.targets:
            total += count_target_names(target)

        return total

    if isinstance(node, ast.AnnAssign):
        return count_target_names(node.target)

    if isinstance(node, ast.AugAssign):
        return count_target_names(node.target)

    return 0


def count_target_names(target):
    """
    Count variable names inside an assignment target.
    """

    # Simple variable:
    # x = 10
    if isinstance(target, ast.Name):
        return 1

    # Tuple/list unpacking:
    # a, b = values
    if isinstance(target, (ast.Tuple, ast.List)):
        total = 0

        for element in target.elts:
            total += count_target_names(element)

        return total

    return 0


def analyze_code(code):
    """
    Alias for calculate_metrics().

    This provides a second intuitive function name that can
    be used by other CodeXray modules if needed.
    """

    return calculate_metrics(code)


if __name__ == "__main__":

    sample_code = """
import math


def calculate_area(radius):
    area = math.pi * radius * radius

    if area > 100:
        print("Large circle")
    else:
        print("Small circle")

    for i in range(3):
        area = area + i

    return area


class Circle:
    pass
"""

    print("=" * 50)
    print("CodeXray Code Metrics Test")
    print("=" * 50)

    metrics = calculate_metrics(sample_code)

    print("\nMetrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")
"""
evaluation.py
-------------
Evaluates CodeXray text similarity using the project dataset.

Dataset structure used by this project:

datasets/
├── metadata.csv
└── original/
    ├── maximum.py
    ├── minimum.py
    ├── factorial.py
    ├── palindrome.py
    ├── prime.py
    │
    └── exact_copy/
        ├── maximum_copy.py
        ├── minimum_copy.py
        ├── factorial_copy.py
        ├── palindrome_copy.py
        └── prime_copy.py
        │
        └── variable_renamed/
            ├── ...
            │
            └── reformatted/
                ├── ...
                │
                └── restructured/
                    ├── ...
                    │
                    └── different_algorithm/
                        ├── ...
                        │
                        └── unrelated/
                            ├── ...

The evaluation calculates:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - False Positives
    - False Negatives
    - True Positives
    - True Negatives

It also creates:

    evaluation/results.csv
    evaluation/confusion_matrix.png
"""

import csv
import sys
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets"

METADATA_FILE = DATASET_ROOT / "metadata.csv"

# Your actual Python files are inside datasets/original.
DATASET_DIR = DATASET_ROOT / "original"

RESULTS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "results.csv"
)

CONFUSION_MATRIX_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "confusion_matrix.png"
)


# =========================================================
# IMPORT TEXT SIMILARITY
# =========================================================

BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from text_similarity import text_similarity


# =========================================================
# DATASET LABELS
# =========================================================

# These categories are considered similar to the original
# implementation.

SIMILAR_CATEGORIES = {
    "original",
    "exact_copy",
    "variable_renamed",
    "reformatted",
    "restructured",
}


# These categories are considered not similar.

NOT_SIMILAR_CATEGORIES = {
    "different_algorithm",
    "unrelated",
}


# =========================================================
# READ SOURCE CODE
# =========================================================

def read_code(file_path):
    """
    Read a Python source-code file.

    Returns:
        str: source code
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# FIND ACTUAL DATASET FILE
# =========================================================

def get_dataset_path(relative_path):
    """
    Convert a metadata path into the actual dataset path.

    The dataset folders are nested like this:

        original/
        original/exact_copy/
        original/exact_copy/variable_renamed/
        original/exact_copy/variable_renamed/reformatted/
        original/exact_copy/variable_renamed/reformatted/restructured/
        original/exact_copy/variable_renamed/reformatted/restructured/different_algorithm/
        original/exact_copy/variable_renamed/reformatted/restructured/different_algorithm/unrelated/
    """

    relative_path = Path(relative_path)

    parts = relative_path.parts

    if len(parts) == 0:
        return DATASET_DIR / relative_path

    category = parts[0]

    # Original files:
    # original/maximum.py
    if category == "original":
        return DATASET_DIR / Path(*parts[1:])

    # Exact-copy files:
    # exact_copy/maximum_copy.py
    if category == "exact_copy":
        return (
            DATASET_DIR
            / "exact_copy"
            / Path(*parts[1:])
        )

    # Variable-renamed files:
    # variable_renamed/maximum_renamed.py
    if category == "variable_renamed":
        return (
            DATASET_DIR
            / "exact_copy"
            / "variable_renamed"
            / Path(*parts[1:])
        )

    # Reformatted files:
    # reformatted/maximum_formatted.py
    if category == "reformatted":
        return (
            DATASET_DIR
            / "exact_copy"
            / "variable_renamed"
            / "reformatted"
            / Path(*parts[1:])
        )

    # Restructured files:
    # restructured/maximum_restructured.py
    if category == "restructured":
        return (
            DATASET_DIR
            / "exact_copy"
            / "variable_renamed"
            / "reformatted"
            / "restructured"
            / Path(*parts[1:])
        )

    # Different-algorithm files:
    # different_algorithm/maximum_builtin.py
    if category == "different_algorithm":
        return (
            DATASET_DIR
            / "exact_copy"
            / "variable_renamed"
            / "reformatted"
            / "restructured"
            / "different_algorithm"
            / Path(*parts[1:])
        )

    # Unrelated files:
    # unrelated/fibonacci.py
    if category == "unrelated":
        return (
            DATASET_DIR
            / "exact_copy"
            / "variable_renamed"
            / "reformatted"
            / "restructured"
            / "different_algorithm"
            / "unrelated"
            / Path(*parts[1:])
        )

    # Fallback
    return DATASET_DIR / relative_path

# =========================================================
# CONVERT CATEGORY TO BINARY LABEL
# =========================================================

def get_actual_label(category):
    """
    Convert dataset category into a binary label.

    Returns:

        1 = Similar
        0 = Not Similar
    """

    if category in SIMILAR_CATEGORIES:
        return 1

    if category in NOT_SIMILAR_CATEGORIES:
        return 0

    raise ValueError(
        f"Unknown dataset category: {category}"
    )


# =========================================================
# CALCULATE CLASSIFICATION METRICS
# =========================================================

def calculate_metrics(actual, predicted):
    """
    Calculate Accuracy, Precision, Recall and F1.

    Returns:
        dictionary containing all metrics.
    """

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # -----------------------------------------------------
    # Count predictions
    # -----------------------------------------------------

    for actual_value, predicted_value in zip(
        actual,
        predicted
    ):

        if actual_value == 1 and predicted_value == 1:

            true_positives += 1

        elif actual_value == 0 and predicted_value == 0:

            true_negatives += 1

        elif actual_value == 0 and predicted_value == 1:

            false_positives += 1

        elif actual_value == 1 and predicted_value == 0:

            false_negatives += 1

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    total = (
        true_positives
        + true_negatives
        + false_positives
        + false_negatives
    )

    if total > 0:

        accuracy = (
            true_positives
            + true_negatives
        ) / total

    else:

        accuracy = 0.0

    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    precision_denominator = (
        true_positives
        + false_positives
    )

    if precision_denominator > 0:

        precision = (
            true_positives
            / precision_denominator
        )

    else:

        precision = 0.0

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    recall_denominator = (
        true_positives
        + false_negatives
    )

    if recall_denominator > 0:

        recall = (
            true_positives
            / recall_denominator
        )

    else:

        recall = 0.0

    # -----------------------------------------------------
    # F1 Score
    # -----------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


# =========================================================
# FIND BEST THRESHOLD
# =========================================================

def find_best_threshold(scores, actual):
    """
    Find a similarity threshold that gives the highest
    F1 score on this dataset.

    Scores greater than or equal to the threshold are
    classified as Similar.
    """

    best_threshold = 0.50
    best_f1 = -1.0

    # Test thresholds from 0.00 to 1.00.

    for number in range(0, 101):

        threshold = number / 100

        predicted = []

        for score in scores:

            if score >= threshold:

                predicted.append(1)

            else:

                predicted.append(0)

        metrics = calculate_metrics(
            actual,
            predicted
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            best_threshold = threshold

    return best_threshold


# =========================================================
# SAVE RESULTS CSV
# =========================================================

def save_results(rows):
    """
    Save individual evaluation results to results.csv.
    """

    fieldnames = [
        "file",
        "problem",
        "category",
        "paired_with",
        "similarity_score",
        "actual",
        "predicted",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(row)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(metrics):
    """
    Save a 2x2 confusion matrix as a PNG image.
    """

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "\nWARNING: matplotlib is not installed."
        )

        print(
            "The evaluation metrics were calculated,"
        )

        print(
            "but confusion_matrix.png could not "
            "be created."
        )

        return

    # -----------------------------------------------------
    # Confusion matrix
    #
    #                  Predicted
    #                Not Sim   Similar
    #
    # Actual
    # Not Sim          TN        FP
    # Similar          FN        TP
    # -----------------------------------------------------

    matrix = [
        [
            metrics["true_negatives"],
            metrics["false_positives"],
        ],
        [
            metrics["false_negatives"],
            metrics["true_positives"],
        ],
    ]

    figure, axis = plt.subplots()

    image = axis.imshow(matrix)

    axis.set_title(
        "CodeXray Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Label"
    )

    axis.set_ylabel(
        "Actual Label"
    )

    axis.set_xticks([0, 1])

    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        [
            "Not Similar",
            "Similar"
        ]
    )

    axis.set_yticklabels(
        [
            "Not Similar",
            "Similar"
        ]
    )

    # Put the numbers inside each square.

    for row_index in range(2):

        for column_index in range(2):

            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index
                    ][
                        column_index
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.colorbar(image)

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150
    )

    plt.close(figure)


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print("=" * 60)

    print(
        "CodeXray Evaluation"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Check metadata file
    # -----------------------------------------------------

    if not METADATA_FILE.exists():

        print("\nERROR:")

        print(
            "Metadata file not found:"
        )

        print(
            METADATA_FILE
        )

        return

    # -----------------------------------------------------
    # Check dataset directory
    # -----------------------------------------------------

    if not DATASET_DIR.exists():

        print("\nERROR:")

        print(
            "Dataset directory not found:"
        )

        print(
            DATASET_DIR
        )

        return

    # -----------------------------------------------------
    # Read metadata
    # -----------------------------------------------------

    metadata_rows = []

    with open(
        METADATA_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            metadata_rows.append(row)

    print(
        f"\nDataset entries found: "
        f"{len(metadata_rows)}"
    )

    if not metadata_rows:

        print(
            "\nERROR: metadata.csv is empty."
        )

        return

    # -----------------------------------------------------
    # Calculate similarity scores
    # -----------------------------------------------------

    scores = []

    actual_labels = []

    evaluation_data = []

    print(
        "\nCalculating similarity scores..."
    )

    for row in metadata_rows:

        relative_file = row["file"]

        relative_pair = row["paired_with"]

        # Find actual files.

        file_path = get_dataset_path(
            relative_file
        )

        pair_path = get_dataset_path(
            relative_pair
        )

        # -------------------------------------------------
        # Check first file
        # -------------------------------------------------

        if not file_path.exists():

            print(
                f"WARNING: File not found: "
                f"{file_path}"
            )

            continue

        # -------------------------------------------------
        # Check paired file
        # -------------------------------------------------

        if not pair_path.exists():

            print(
                f"WARNING: Paired file not found: "
                f"{pair_path}"
            )

            continue

        # -------------------------------------------------
        # Read both files
        # -------------------------------------------------

        code1 = read_code(
            file_path
        )

        code2 = read_code(
            pair_path
        )

        # -------------------------------------------------
        # Calculate similarity
        # -------------------------------------------------

        score = text_similarity(
            code1,
            code2
        )

        # -------------------------------------------------
        # Get actual label
        # -------------------------------------------------

        actual = get_actual_label(
            row["category"]
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        scores.append(score)

        actual_labels.append(actual)

        evaluation_data.append({
            "file": relative_file,
            "problem": row["problem"],
            "category": row["category"],
            "paired_with": relative_pair,
            "score": score,
            "actual": actual,
        })

    # -----------------------------------------------------
    # Check valid comparisons
    # -----------------------------------------------------

    if not evaluation_data:

        print(
            "\nERROR: No valid dataset pairs "
            "were found."
        )

        print(
            "\nExpected dataset directory:"
        )

        print(
            DATASET_DIR
        )

        return

    print(
        f"Valid comparisons: "
        f"{len(evaluation_data)}"
    )

    # -----------------------------------------------------
    # Find threshold
    # -----------------------------------------------------

    threshold = find_best_threshold(
        scores,
        actual_labels
    )

    print(
        f"\nSelected similarity threshold: "
        f"{threshold:.2f}"
    )

    print(
        "(Threshold selected using this "
        "evaluation dataset.)"
    )

    # -----------------------------------------------------
    # Make predictions
    # -----------------------------------------------------

    predicted_labels = []

    result_rows = []

    for item in evaluation_data:

        score = item["score"]

        if score >= threshold:

            predicted = 1

        else:

            predicted = 0

        predicted_labels.append(
            predicted
        )

        result_rows.append({
            "file": item["file"],
            "problem": item["problem"],
            "category": item["category"],
            "paired_with": item["paired_with"],
            "similarity_score": (
                f"{score:.4f}"
            ),
            "actual": (
                "Similar"
                if item["actual"] == 1
                else "Not Similar"
            ),
            "predicted": (
                "Similar"
                if predicted == 1
                else "Not Similar"
            ),
        })

    # -----------------------------------------------------
    # Calculate final metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(
        actual_labels,
        predicted_labels
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    save_results(
        result_rows
    )

    save_confusion_matrix(
        metrics
    )

    # -----------------------------------------------------
    # Print final results
    # -----------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "Evaluation Results"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy:          "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Precision:         "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall:            "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"F1 Score:          "
        f"{metrics['f1']:.2%}"
    )

    print(
        f"\nFalse Positives:   "
        f"{metrics['false_positives']}"
    )

    print(
        f"False Negatives:   "
        f"{metrics['false_negatives']}"
    )

    print(
        f"True Positives:    "
        f"{metrics['true_positives']}"
    )

    print(
        f"True Negatives:    "
        f"{metrics['true_negatives']}"
    )

    # -----------------------------------------------------
    # Output files
    # -----------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "Files Created"
    )

    print(
        "=" * 60
    )

    print(
        f"\nResults CSV:"
    )

    print(
        RESULTS_FILE
    )

    print(
        f"\nConfusion Matrix:"
    )

    print(
        CONFUSION_MATRIX_FILE
    )

    print(
        "\nEvaluation complete."
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    main()
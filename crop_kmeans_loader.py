import os
import pandas as pd
import numpy as np


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# CROP YIELD PREPROCESSED DATASET
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "crop_yield_preprocessed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n========================================")
    print("K-MEANS DATA LOADER")
    print("========================================")

    print(
        "Loading dataset from:"
    )

    print(DATASET_PATH)


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "Crop Yield preprocessed dataset "
            "not found:\n"
            + DATASET_PATH
        )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    data = pd.read_csv(
        DATASET_PATH
    )


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )


    print(
        "Dataset shape:",
        data.shape
    )


    print(
        "Columns:"
    )

    print(
        data.columns.tolist()
    )


    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    data = data.drop_duplicates()


    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Select numerical columns
    # --------------------------------------------------------

    X = data.select_dtypes(
        include=np.number
    ).copy()


    # --------------------------------------------------------
    # Remove Yield from clustering
    # --------------------------------------------------------

    yield_columns = [

        "Yield",
        "yield",
        "Yield_tons",
        "Yield_Production",
        "Crop_Yield"

    ]


    for column in yield_columns:

        if column in X.columns:

            print(
                f"Removing target column: {column}"
            )

            X = X.drop(
                column,
                axis=1
            )

            break


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if X.isnull().sum().sum() > 0:

        print(
            "Missing values found."
        )

        X = X.fillna(
            X.median()
        )


    # --------------------------------------------------------
    # Remaining missing values
    # --------------------------------------------------------

    X = X.fillna(0)


    # --------------------------------------------------------
    # Check minimum features
    # --------------------------------------------------------

    if X.shape[1] < 2:

        raise ValueError(
            "K-Means requires at least "
            "2 numerical features."
        )


    print(
        "K-Means input shape:",
        X.shape
    )


    print(
        "K-Means features:"
    )

    print(
        X.columns.tolist()
    )


    print(
        "========================================"
    )


    return X
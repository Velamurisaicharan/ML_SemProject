import os
import io
import base64

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "crop_yield.csv"
)

PROCESSED_PATH = os.path.join(
    BASE_DIR,
    "crop_yield_preprocessed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_crop_dataset():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


# ============================================================
# GRAPH TO BASE64
# ============================================================

def figure_to_base64(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def run_crop_preprocessing():

    # ========================================================
    # 1. LOAD
    # ========================================================

    original_df = load_crop_dataset()

    df = original_df.copy()


    # ========================================================
    # 2. BASIC INFORMATION
    # ========================================================

    original_rows = int(
        df.shape[0]
    )

    original_columns = int(
        df.shape[1]
    )

    original_missing = int(
        df.isna().sum().sum()
    )

    original_duplicates = int(
        df.duplicated().sum()
    )


    # ========================================================
    # 3. CLEAN CATEGORICAL TEXT
    # ========================================================

    categorical_columns = [
        "Crop",
        "Season",
        "State"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
            )


    # ========================================================
    # 4. REMOVE DUPLICATES
    # ========================================================

    duplicates_removed = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()


    # ========================================================
    # 5. MISSING VALUE INFORMATION
    # ========================================================

    missing_table = []

    for column in df.columns:

        count = int(
            df[column].isna().sum()
        )

        percentage = (
            count / len(df) * 100
        )

        missing_table.append({

            "column": column,

            "count": count,

            "percentage":
                round(
                    percentage,
                    2
                )

        })


    # ========================================================
    # 6. TARGET
    # ========================================================

    target_column = "Yield"


    if target_column not in df.columns:

        raise ValueError(
            "Yield column was not found in dataset."
        )


    # ========================================================
    # 7. FEATURES
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # ========================================================
    # 8. TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )


    # ========================================================
    # 9. CATEGORICAL FEATURES
    # ========================================================

    categorical_features = [

        "Crop",
        "Season",
        "State"

    ]

    categorical_features = [

        column

        for column in categorical_features

        if column in X_train.columns

    ]


    # ========================================================
    # 10. NUMERICAL FEATURES
    # ========================================================

    numerical_features = [

        "Crop_Year",
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"

    ]

    numerical_features = [

        column

        for column in numerical_features

        if column in X_train.columns

    ]


    # ========================================================
    # 11. ONE-HOT ENCODING
    # ========================================================

    try:

        encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse_output=False

        )

    except TypeError:

        encoder = OneHotEncoder(

            handle_unknown="ignore",

            sparse=False

        )


    if categorical_features:

        train_encoded = encoder.fit_transform(

            X_train[
                categorical_features
            ]

        )

        test_encoded = encoder.transform(

            X_test[
                categorical_features
            ]

        )

        encoded_columns = (

            encoder
            .get_feature_names_out(
                categorical_features
            )
            .tolist()

        )


        X_train_cat = pd.DataFrame(

            train_encoded,

            columns=encoded_columns,

            index=X_train.index

        )


        X_test_cat = pd.DataFrame(

            test_encoded,

            columns=encoded_columns,

            index=X_test.index

        )

    else:

        encoded_columns = []

        X_train_cat = pd.DataFrame(
            index=X_train.index
        )

        X_test_cat = pd.DataFrame(
            index=X_test.index
        )


    # ========================================================
    # 12. OUTLIER DETECTION
    # ========================================================

    outlier_details = []

    X_train_clean = X_train.copy()

    X_test_clean = X_test.copy()


    for column in numerical_features:

        q1 = X_train_clean[
            column
        ].quantile(0.25)

        q3 = X_train_clean[
            column
        ].quantile(0.75)

        iqr = q3 - q1


        if iqr == 0 or pd.isna(iqr):

            continue


        lower_limit = (
            q1 - 1.5 * iqr
        )

        upper_limit = (
            q3 + 1.5 * iqr
        )


        train_mask = (

            (X_train_clean[column] < lower_limit)

            |

            (X_train_clean[column] > upper_limit)

        )


        test_mask = (

            (X_test_clean[column] < lower_limit)

            |

            (X_test_clean[column] > upper_limit)

        )


        train_count = int(
            train_mask.sum()
        )

        test_count = int(
            test_mask.sum()
        )


        if train_count > 0:

            outlier_details.append({

                "column": column,

                "train_count":
                    train_count,

                "test_count":
                    test_count,

                "percentage":
                    round(
                        train_count
                        /
                        len(X_train_clean)
                        *
                        100,
                        2
                    )

            })


        # IQR clipping
        # Only INPUT features are clipped.
        # Yield is NOT clipped because it is the target.

        X_train_clean[column] = (
            X_train_clean[column]
            .clip(
                lower_limit,
                upper_limit
            )
        )

        X_test_clean[column] = (
            X_test_clean[column]
            .clip(
                lower_limit,
                upper_limit
            )
        )


    # ========================================================
    # 13. FEATURE SCALING
    # ========================================================

    scaler = StandardScaler()


    if numerical_features:

        train_scaled_values = (
            scaler.fit_transform(
                X_train_clean[
                    numerical_features
                ]
            )
        )

        test_scaled_values = (
            scaler.transform(
                X_test_clean[
                    numerical_features
                ]
            )
        )


        X_train_num = pd.DataFrame(

            train_scaled_values,

            columns=numerical_features,

            index=X_train.index

        )


        X_test_num = pd.DataFrame(

            test_scaled_values,

            columns=numerical_features,

            index=X_test.index

        )

    else:

        X_train_num = pd.DataFrame(
            index=X_train.index
        )

        X_test_num = pd.DataFrame(
            index=X_test.index
        )


    # ========================================================
    # 14. FINAL DATASET
    # ========================================================

    X_train_final = pd.concat(

        [
            X_train_num,
            X_train_cat
        ],

        axis=1

    )


    X_test_final = pd.concat(

        [
            X_test_num,
            X_test_cat
        ],

        axis=1

    )


    # Add target

    X_train_final[
        "Yield"
    ] = y_train


    X_test_final[
        "Yield"
    ] = y_test


    # Combine

    final_df = pd.concat(

        [
            X_train_final,
            X_test_final
        ],

        axis=0

    ).sort_index()


    # ========================================================
    # 15. SAVE
    # ========================================================

    final_df.to_csv(

        PROCESSED_PATH,

        index=False

    )


    # ========================================================
    # 16. SCALING STATISTICS
    # ========================================================

    scaling_statistics = []


    for column in numerical_features:

        before_series = X_train[
            column
        ]

        after_series = X_train_num[
            column
        ]


        scaling_statistics.append({

            "feature": column,

            "before_mean":
                round(
                    float(
                        before_series.mean()
                    ),
                    3
                ),

            "before_std":
                round(
                    float(
                        before_series.std()
                    ),
                    3
                ),

            "after_mean":
                round(
                    float(
                        after_series.mean()
                    ),
                    4
                ),

            "after_std":
                round(
                    float(
                        after_series.std()
                    ),
                    4
                )

        })


    # ========================================================
    # 17. BEFORE / AFTER SCALING GRAPH
    # ========================================================

    scaling_plot = None


    if numerical_features:

        feature = numerical_features[0]


        fig = plt.figure(
            figsize=(9, 5)
        )


        plt.hist(

            X_train[feature],

            bins=30,

            alpha=0.7,

            label="Before Scaling"

        )


        plt.hist(

            X_train_num[feature],

            bins=30,

            alpha=0.7,

            label="After Scaling"

        )


        plt.title(
            f"{feature}: Before vs After Scaling"
        )


        plt.xlabel(feature)

        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()


        scaling_plot = figure_to_base64(
            fig
        )


    # ========================================================
    # 18. OUTLIER GRAPH
    # ========================================================

    outlier_plot = None


    if outlier_details:

        names = [

            item["column"]

            for item in outlier_details

        ]

        values = [

            item["train_count"]

            for item in outlier_details

        ]


        fig = plt.figure(
            figsize=(9, 5)
        )


        plt.bar(
            names,
            values
        )


        plt.title(
            "Input Feature Outliers"
        )


        plt.ylabel(
            "Outlier Count"
        )


        plt.xticks(
            rotation=30,
            ha="right"
        )


        plt.tight_layout()


        outlier_plot = figure_to_base64(
            fig
        )


    # ========================================================
    # 19. ENCODING GRAPH
    # ========================================================

    encoding_plot = None


    encoding_values = [

        len(encoded_columns),

        len(categorical_features),

        len(numerical_features)

    ]


    encoding_names = [

        "Encoded Features",

        "Categorical",

        "Numerical"

    ]


    fig = plt.figure(
        figsize=(8, 5)
    )


    plt.bar(

        encoding_names,

        encoding_values

    )


    plt.title(
        "Feature Transformation Summary"
    )


    plt.ylabel(
        "Number of Features"
    )


    plt.tight_layout()


    encoding_plot = figure_to_base64(
        fig
    )


    # ========================================================
    # 20. TARGET INFORMATION
    # ========================================================

    target_statistics = {

        "mean":
            round(
                float(y.mean()),
                3
            ),

        "std":
            round(
                float(y.std()),
                3
            ),

        "min":
            round(
                float(y.min()),
                3
            ),

        "max":
            round(
                float(y.max()),
                3
            )

    }


    # ========================================================
    # 21. FINAL PREVIEW
    # ========================================================

    preview = final_df.head(10).copy()


    # ========================================================
    # 22. RETURN
    # ========================================================

    return {

        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "original_missing":
            original_missing,

        "original_duplicates":
            original_duplicates,

        "duplicates_removed":
            duplicates_removed,

        "final_rows":
            int(final_df.shape[0]),

        "final_columns":
            int(final_df.shape[1]),

        "final_missing":
            int(final_df.isna().sum().sum()),

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "categorical_features":
            categorical_features,

        "numerical_features":
            numerical_features,

        "encoded_columns":
            encoded_columns,

        "outlier_details":
            outlier_details,

        "scaling_statistics":
            scaling_statistics,

        "target_statistics":
            target_statistics,

        "missing_table":
            missing_table,

        "preview_columns":
            preview.columns.tolist(),

        "preview":
            preview.to_dict(
                orient="records"
            ),

        "scaling_plot":
            scaling_plot,

        "outlier_plot":
            outlier_plot,

        "encoding_plot":
            encoding_plot

    }
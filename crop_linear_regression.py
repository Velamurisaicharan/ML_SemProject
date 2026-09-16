import os
import io
import base64

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "crop_yield.csv"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "crop_linear_regression_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_crop_data():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Crop Yield dataset not found:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if df.empty:

        raise ValueError(
            "Crop Yield dataset is empty."
        )

    return df


# ============================================================
# FIGURE TO BASE64
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
# MAIN LINEAR REGRESSION
# ============================================================

def run_crop_linear_regression():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_crop_data()


    # ========================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # 3. TARGET
    # ========================================================

    target_column = "Yield"


    if target_column not in df.columns:

        raise ValueError(
            "Yield column was not found in the dataset."
        )


    # ========================================================
    # 4. REMOVE ROWS WITH MISSING TARGET
    # ========================================================

    original_rows = len(df)

    df = df.dropna(
        subset=[target_column]
    ).copy()

    removed_target_rows = (
        original_rows - len(df)
    )


    # ========================================================
    # 5. FEATURES
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )


    # Remove rows where Yield could not be converted

    valid_target = y.notna()

    X = X.loc[
        valid_target
    ].copy()

    y = y.loc[
        valid_target
    ].copy()


    if len(X) < 10:

        raise ValueError(
            "Not enough valid rows for Linear Regression."
        )


    # ========================================================
    # 6. DEFINE FEATURE TYPES
    # ========================================================

    categorical_features = [

        "Crop",
        "Season",
        "State"

    ]

    categorical_features = [

        column

        for column in categorical_features

        if column in X.columns

    ]


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

        if column in X.columns

    ]


    # ========================================================
    # 7. TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )


    # ========================================================
    # 8. NUMERICAL PIPELINE
    # ========================================================

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )

            ),

            (
                "scaler",

                StandardScaler()

            )

        ]

    )


    # ========================================================
    # 9. CATEGORICAL ENCODER
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


    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )

            ),

            (
                "encoder",

                encoder

            )

        ]

    )


    # ========================================================
    # 10. COLUMN TRANSFORMER
    # ========================================================

    transformers = []


    if numerical_features:

        transformers.append(

            (
                "numeric",

                numerical_pipeline,

                numerical_features

            )

        )


    if categorical_features:

        transformers.append(

            (
                "categorical",

                categorical_pipeline,

                categorical_features

            )

        )


    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop"

    )


    # ========================================================
    # 11. LINEAR REGRESSION MODEL
    # ========================================================

    regression = LinearRegression()


    # ========================================================
    # 12. COMPLETE PIPELINE
    # ========================================================

    model = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor

            ),

            (
                "regressor",

                regression

            )

        ]

    )


    # ========================================================
    # 13. TRAIN MODEL
    # ========================================================

    model.fit(

        X_train,

        y_train

    )


    # ========================================================
    # 14. PREDICTIONS
    # ========================================================

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )


    # ========================================================
    # 15. METRICS
    # ========================================================

    train_r2 = r2_score(

        y_train,

        y_train_pred

    )

    test_r2 = r2_score(

        y_test,

        y_test_pred

    )


    mae = mean_absolute_error(

        y_test,

        y_test_pred

    )


    mse = mean_squared_error(

        y_test,

        y_test_pred

    )


    rmse = np.sqrt(
        mse
    )


    # ========================================================
    # 16. ADJUSTED R2
    # ========================================================

    try:

        transformed_train = (

            model
            .named_steps["preprocessor"]
            .transform(X_train)

        )

        n = transformed_train.shape[0]

        p = transformed_train.shape[1]


        if n > p + 1:

            adjusted_r2 = (

                1
                -
                (
                    (1 - test_r2)
                    *
                    (n - 1)
                    /
                    (n - p - 1)
                )

            )

        else:

            adjusted_r2 = test_r2

    except Exception:

        adjusted_r2 = test_r2


    # ========================================================
    # 17. FEATURE NAMES
    # ========================================================

    fitted_preprocessor = (

        model
        .named_steps["preprocessor"]

    )


    try:

        feature_names = (

            fitted_preprocessor
            .get_feature_names_out()
            .tolist()

        )

    except Exception:

        feature_names = [

            f"Feature_{i+1}"

            for i in range(

                len(
                    regression.coef_
                )

            )

        ]


    coefficients = regression.coef_


    # ========================================================
    # 18. COEFFICIENT TABLE
    # ========================================================

    coefficient_data = []


    for feature, coefficient in zip(

        feature_names,

        coefficients

    ):

        coefficient_data.append({

            "feature":
                str(feature),

            "coefficient":
                round(
                    float(coefficient),
                    6
                ),

            "absolute":
                round(
                    abs(float(coefficient)),
                    6
                )

        })


    coefficient_data.sort(

        key=lambda item:
            item["absolute"],

        reverse=True

    )


    # ========================================================
    # 19. ACTUAL VS PREDICTED GRAPH
    # ========================================================

    fig = plt.figure(
        figsize=(8, 6)
    )


    plt.scatter(

        y_test,

        y_test_pred,

        alpha=0.5

    )


    min_value = min(

        y_test.min(),

        y_test_pred.min()

    )


    max_value = max(

        y_test.max(),

        y_test_pred.max()

    )


    plt.plot(

        [min_value, max_value],

        [min_value, max_value],

        linestyle="--"

    )


    plt.xlabel(
        "Actual Yield"
    )


    plt.ylabel(
        "Predicted Yield"
    )


    plt.title(
        "Actual vs Predicted Crop Yield"
    )


    plt.tight_layout()


    actual_predicted_plot = (
        figure_to_base64(fig)
    )


    # ========================================================
    # 20. RESIDUAL PLOT
    # ========================================================

    residuals = (
        y_test.values
        -
        y_test_pred
    )


    fig = plt.figure(
        figsize=(8, 6)
    )


    plt.scatter(

        y_test_pred,

        residuals,

        alpha=0.5

    )


    plt.axhline(
        y=0,
        linestyle="--"
    )


    plt.xlabel(
        "Predicted Yield"
    )


    plt.ylabel(
        "Residual"
    )


    plt.title(
        "Residual Analysis"
    )


    plt.tight_layout()


    residual_plot = (
        figure_to_base64(fig)
    )


    # ========================================================
    # 21. COEFFICIENT GRAPH
    # ========================================================

    top_coefficients = (
        coefficient_data[:15]
    )


    coefficient_plot = None


    if top_coefficients:

        names = [

            item["feature"]

            for item in top_coefficients

        ]

        values = [

            item["coefficient"]

            for item in top_coefficients

        ]


        fig = plt.figure(
            figsize=(10, 6)
        )


        plt.barh(

            names[::-1],

            values[::-1]

        )


        plt.xlabel(
            "Coefficient"
        )


        plt.ylabel(
            "Feature"
        )


        plt.title(
            "Top Linear Regression Coefficients"
        )


        plt.tight_layout()


        coefficient_plot = (
            figure_to_base64(fig)
        )


    # ========================================================
    # 22. PREDICTION CSV
    # ========================================================

    prediction_df = X_test.copy()


    prediction_df[
        "Actual_Yield"
    ] = y_test.values


    prediction_df[
        "Predicted_Yield"
    ] = y_test_pred


    prediction_df[
        "Residual"
    ] = residuals


    prediction_df[
        "Absolute_Error"
    ] = np.abs(
        residuals
    )


    prediction_df.to_csv(

        PREDICTION_PATH,

        index=False

    )


    # ========================================================
    # 23. SAMPLE PREDICTIONS
    # ========================================================

    sample_predictions = []

    sample_count = min(
        10,
        len(y_test)
    )


    for i in range(
        sample_count
    ):

        sample_predictions.append({

            "actual":
                round(
                    float(
                        y_test.iloc[i]
                    ),
                    4
                ),

            "predicted":
                round(
                    float(
                        y_test_pred[i]
                    ),
                    4
                ),

            "error":
                round(
                    float(
                        residuals[i]
                    ),
                    4
                )

        })


    # ========================================================
    # 24. RETURN RESULTS
    # ========================================================

    return {

        "target":
            target_column,

        "total_rows":
            int(len(X)),

        "original_rows":
            int(original_rows),

        "removed_target_rows":
            int(removed_target_rows),

        "total_features":
            int(
                len(
                    numerical_features
                )
                +
                len(
                    categorical_features
                )
            ),

        "numerical_features":
            numerical_features,

        "categorical_features":
            categorical_features,

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "transformed_features":
            int(
                len(feature_names)
            ),

        "train_r2":
            round(
                float(train_r2),
                4
            ),

        "test_r2":
            round(
                float(test_r2),
                4
            ),

        "adjusted_r2":
            round(
                float(adjusted_r2),
                4
            ),

        "mae":
            round(
                float(mae),
                4
            ),

        "mse":
            round(
                float(mse),
                4
            ),

        "rmse":
            round(
                float(rmse),
                4
            ),

        "intercept":
            round(
                float(
                    regression.intercept_
                ),
                6
            ),

        "coefficient_data":
            coefficient_data,

        "sample_predictions":
            sample_predictions,

        "actual_predicted_plot":
            actual_predicted_plot,

        "residual_plot":
            residual_plot,

        "coefficient_plot":
            coefficient_plot

    }
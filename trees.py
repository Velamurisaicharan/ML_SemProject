import os
import io
import base64

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import (
    DecisionTreeRegressor,
    plot_tree
)

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
    r"D:\AI_ML\ML_SemProject\crop_yield_preprocessed.csv"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "tree_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_tree_data():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Processed Crop Yield dataset not found:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    if df.empty:

        raise ValueError(
            "Dataset is empty."
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
# DECISION TREE REGRESSION
# ============================================================

def run_decision_tree():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_tree_data()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # 2. TARGET
    # --------------------------------------------------------

    target = "Yield"

    if target not in df.columns:

        raise ValueError(
            "Target column 'Yield' was not found."
        )


    # --------------------------------------------------------
    # 3. CONVERT TARGET TO NUMERIC
    # --------------------------------------------------------

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 4. REMOVE INVALID TARGET ROWS
    # --------------------------------------------------------

    original_rows = len(df)

    df = df.dropna(
        subset=[target]
    ).copy()


    # --------------------------------------------------------
    # 5. X AND Y
    # --------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]


    # --------------------------------------------------------
    # 6. KEEP NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()


    if len(numeric_columns) == 0:

        raise ValueError(
            "No numeric features available "
            "for Decision Tree."
        )


    X = X[numeric_columns]


    # --------------------------------------------------------
    # 7. REMOVE INFINITE VALUES
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # 8. MISSING VALUE IMPUTATION
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_processed = imputer.fit_transform(
        X
    )


    # --------------------------------------------------------
    # 9. TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X_processed,

        y,

        test_size=0.20,

        random_state=42

    )


    # --------------------------------------------------------
    # 10. DECISION TREE
    # --------------------------------------------------------

    tree_model = DecisionTreeRegressor(

        criterion="squared_error",

        max_depth=8,

        min_samples_split=10,

        min_samples_leaf=5,

        random_state=42

    )


    # --------------------------------------------------------
    # 11. TRAIN
    # --------------------------------------------------------

    tree_model.fit(

        X_train,

        y_train

    )


    # --------------------------------------------------------
    # 12. PREDICT
    # --------------------------------------------------------

    y_train_pred = tree_model.predict(
        X_train
    )

    y_test_pred = tree_model.predict(
        X_test
    )


    # --------------------------------------------------------
    # 13. METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 14. RESIDUALS
    # --------------------------------------------------------

    residuals = (
        y_test.values
        -
        y_test_pred
    )


    # ========================================================
    # TREE INFORMATION
    # ========================================================

    tree_depth = tree_model.get_depth()

    leaf_count = tree_model.get_n_leaves()

    node_count = tree_model.tree_.node_count


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importances = (
        tree_model.feature_importances_
    )


    importance_data = []


    for feature, importance in zip(

        numeric_columns,

        importances

    ):

        importance_data.append({

            "feature":
                feature,

            "importance":
                round(
                    float(importance),
                    6
                )

        })


    importance_data.sort(

        key=lambda x:
            x["importance"],

        reverse=True

    )


    # ========================================================
    # TOP 15 FEATURES GRAPH
    # ========================================================

    top_features = (
        importance_data[:15]
    )


    feature_importance_plot = None


    if top_features:

        names = [

            item["feature"]

            for item in top_features

        ]

        values = [

            item["importance"]

            for item in top_features

        ]


        fig = plt.figure(
            figsize=(10, 7)
        )


        plt.barh(

            names[::-1],

            values[::-1]

        )


        plt.xlabel(
            "Importance"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Top Decision Tree Feature Importance"
        )


        plt.tight_layout()


        feature_importance_plot = (
            figure_to_base64(fig)
        )


    # ========================================================
    # TREE VISUALIZATION
    # ========================================================

    fig = plt.figure(
        figsize=(24, 12)
    )


    plot_tree(

        tree_model,

        feature_names=numeric_columns,

        filled=True,

        rounded=True,

        max_depth=4,

        fontsize=7

    )


    plt.title(
        "Decision Tree Regression Structure"
    )


    plt.tight_layout()


    tree_plot = (
        figure_to_base64(fig)
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )


    plt.scatter(

        y_test,

        y_test_pred,

        alpha=0.5

    )


    minimum = min(

        y_test.min(),

        y_test_pred.min()

    )


    maximum = max(

        y_test.max(),

        y_test_pred.max()

    )


    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

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
    # RESIDUAL GRAPH
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )


    plt.scatter(

        y_test_pred,

        residuals,

        alpha=0.5

    )


    plt.axhline(

        0,

        linestyle="--"

    )


    plt.xlabel(
        "Predicted Yield"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Decision Tree Residual Analysis"
    )


    plt.tight_layout()


    residual_plot = (
        figure_to_base64(fig)
    )


    # ========================================================
    # PREDICTION CSV
    # ========================================================

    prediction_df = pd.DataFrame({

        "Actual_Yield":
            y_test.values,

        "Predicted_Yield":
            y_test_pred,

        "Residual":
            residuals,

        "Absolute_Error":
            np.abs(residuals)

    })


    prediction_df.to_csv(

        PREDICTION_PATH,

        index=False

    )


    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    sample_predictions = []


    sample_count = min(

        15,

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
    # RETURN RESULTS
    # ========================================================

    return {

        "target":
            target,

        "original_rows":
            original_rows,

        "rows_used":
            len(df),

        "feature_count":
            len(numeric_columns),

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "tree_depth":
            tree_depth,

        "leaf_count":
            leaf_count,

        "node_count":
            node_count,

        "criterion":
            "Squared Error (MSE)",

        "max_depth":
            8,

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

        "features":
            numeric_columns,

        "importance_data":
            importance_data,

        "sample_predictions":
            sample_predictions,

        "tree_plot":
            tree_plot,

        "feature_importance_plot":
            feature_importance_plot,

        "actual_predicted_plot":
            actual_predicted_plot,

        "residual_plot":
            residual_plot

    }
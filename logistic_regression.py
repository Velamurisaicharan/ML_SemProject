import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "D:\AI_ML\ML_SemProject\crop_yield_preprocessed.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

TARGET = "Yield"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "processed_dataset.csv was not found.\n"
            "Place the dataset in the same folder as "
            "logistic_regression.py."
        )

    df = pd.read_csv(DATASET_PATH)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data():

    df = load_dataset()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    data = df[
        required_columns
    ].copy()

    data = data.dropna()

    # --------------------------------------------------------
    # Create binary target
    # --------------------------------------------------------

    median_yield = data[TARGET].median()

    data["Yield_Class"] = (
        data[TARGET] > median_yield
    ).astype(int)

    X = data[FEATURES]

    y = data["Yield_Class"]

    return (
        data,
        X,
        y,
        median_yield
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    (
        data,
        X,
        y,
        median_yield
    ) = prepare_data()


    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    # --------------------------------------------------------
    # Standard Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_test_scaled = (
        scaler.transform(X_test)
    )


    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(

        max_iter=2000,

        random_state=42

    )


    model.fit(

        X_train_scaled,

        y_train

    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )

    y_probability = (
        model
        .predict_proba(
            X_test_scaled
        )[:, 1]
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    coefficients = []

    for feature, coefficient in zip(
        FEATURES,
        model.coef_[0]
    ):

        coefficients.append({

            "feature":
                feature,

            "coefficient":
                float(coefficient)

        })


    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    prediction_file = os.path.join(

        BASE_DIR,

        "logistic_yield_predictions.csv"

    )


    prediction_data = X_test.copy()

    prediction_data[
        "Actual_Yield_Class"
    ] = y_test.values

    prediction_data[
        "Predicted_Yield_Class"
    ] = y_pred

    prediction_data[
        "High_Yield_Probability"
    ] = y_probability


    prediction_data[
        "Actual_Yield_Label"
    ] = prediction_data[
        "Actual_Yield_Class"
    ].map({

        0: "Low Yield",

        1: "High Yield"

    })


    prediction_data[
        "Predicted_Yield_Label"
    ] = prediction_data[
        "Predicted_Yield_Class"
    ].map({

        0: "Low Yield",

        1: "High Yield"

    })


    prediction_data.to_csv(

        prediction_file,

        index=False

    )


    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        "model":
            model,

        "scaler":
            scaler,

        "features":
            FEATURES,

        "median_yield":
            median_yield,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "auc":
            auc,

        "confusion_matrix":
            cm.tolist(),

        "coefficients":
            coefficients,

        "training_samples":
            len(X_train),

        "testing_samples":
            len(X_test),

        "low_yield_count":
            int((y == 0).sum()),

        "high_yield_count":
            int((y == 1).sum()),

        "prediction_file":
            prediction_file

    }


# ============================================================
# PREDICT NEW DATA
# ============================================================

def predict_yield(
    crop_year,
    area,
    rainfall,
    fertilizer,
    pesticide
):

    results = train_model()

    model = results["model"]

    scaler = results["scaler"]


    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(

        [[
            crop_year,
            area,
            rainfall,
            fertilizer,
            pesticide
        ]],

        columns=FEATURES

    )


    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = (

        model
        .predict_proba(
            input_scaled
        )[0][1]

    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    predicted_class = (

        1
        if probability >= 0.50
        else 0

    )


    label = (

        "High Yield"
        if predicted_class == 1
        else "Low Yield"

    )


    return {

        "class":
            predicted_class,

        "label":
            label,

        "probability":
            probability,

        "low_probability":
            1 - probability

    }


# ============================================================
# TEST FROM TERMINAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       LOGISTIC REGRESSION - YIELD CLASSIFICATION")
    print("=" * 60)


    try:

        results = train_model()


        print()

        print(
            "Median Yield Threshold:",
            round(
                results["median_yield"],
                4
            )
        )

        print()

        print(
            "Training Samples:",
            results["training_samples"]
        )

        print(
            "Testing Samples:",
            results["testing_samples"]
        )

        print()

        print(
            "Low Yield Samples:",
            results["low_yield_count"]
        )

        print(
            "High Yield Samples:",
            results["high_yield_count"]
        )

        print()

        print("-" * 60)

        print(
            "Accuracy :",
            f"{results['accuracy']:.4f}"
        )

        print(
            "Precision:",
            f"{results['precision']:.4f}"
        )

        print(
            "Recall   :",
            f"{results['recall']:.4f}"
        )

        print(
            "F1 Score :",
            f"{results['f1']:.4f}"
        )

        print(
            "ROC-AUC  :",
            f"{results['auc']:.4f}"
        )

        print("-" * 60)

        print()

        print("Confusion Matrix:")

        print(
            np.array(
                results["confusion_matrix"]
            )
        )

        print()

        print("Feature Coefficients:")

        for item in results["coefficients"]:

            print(

                f"{item['feature']:25s}"
                f"{item['coefficient']:.6f}"

            )

        print()

        print(
            "Predictions saved to:"
        )

        print(
            results["prediction_file"]
        )

        print()

        print("=" * 60)

    except Exception as e:

        print()

        print(
            "ERROR:",
            type(e).__name__,
            str(e)
        )

        print()
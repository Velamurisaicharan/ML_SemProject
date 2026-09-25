import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import os

from flask import (
    Flask,
    render_template,
    send_file,
    request
)

from load_data import (
    load_data,
    get_data_summary
)

from crop_eda import run_eda

from crop_preprocessing import (
    run_crop_preprocessing
)
from crop_linear_regression import run_crop_linear_regression
from logistic_regression import train_model, predict_yield
from trees import run_decision_tree
from crop_kmeans import run_kmeans


app = Flask(__name__)
import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="dashboard"
    )
@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:
        df = load_data()
        summary = get_data_summary(df)

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        error=error
    )



# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except Exception as e:

        error = str(e)

    return render_template(

        "eda.html",

        active="eda",

        results=results,

        error=error

    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():

    error = None
    results = None

    try:

        results = run_crop_preprocessing()

    except Exception as e:

        error = (
            f"{type(e).__name__}: {str(e)}"
        )

    return render_template(

        "preprocessing.html",

        active="preprocessing",

        results=results,

        error=error

    )


# ============================================================
# DOWNLOAD
# ============================================================

@app.route("/download-crop-processed")
def download_crop_processed():

    return send_file(

        "crop_yield_preprocessed.csv",

        as_attachment=True,

        download_name=
        "crop_yield_preprocessed.csv"

    )
# ============================================================
# CROP YIELD - LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression_page():

    error = None
    results = None

    try:

        results = run_crop_linear_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = (
            f"{type(e).__name__}: {str(e)}"
        )

    return render_template(

        "linear_regression.html",

        active="linear-regression",

        results=results,

        error=error

    )
# ============================================================
# CROP YIELD - LINEAR REGRESSION
# ============================================================


@app.route("/download-crop-linear-regression")
def download_crop_linear_regression():

    prediction_file = os.path.join(
        BASE_DIR,
        "crop_linear_regression_predictions.csv"
    )

    if not os.path.exists(prediction_file):

        return (
            "Crop Yield predictions have not "
            "been generated yet.",
            404
        )

    return send_file(
        prediction_file,
        as_attachment=True,
        download_name="crop_linear_regression_predictions.csv"
    )
# ============================================================
# LOGISTIC REGRESSION - YIELD CLASSIFICATION
# ============================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    error = None
    results = None

    try:

        results = train_model()

    except Exception as e:

        error = (
            f"{type(e).__name__}: "
            f"{str(e)}"
        )


    return render_template(

        "logistic_regression.html",

        active="logistic-regression",

        results=results,

        error=error

    )


# ============================================================
# YIELD PREDICTION
# ============================================================

@app.route(
    "/predict-yield",
    methods=["POST"]
)
def predict_yield_page():

    try:

        crop_year = float(
            request.form["crop_year"]
        )

        area = float(
            request.form["area"]
        )

        rainfall = float(
            request.form["rainfall"]
        )

        fertilizer = float(
            request.form["fertilizer"]
        )

        pesticide = float(
            request.form["pesticide"]
        )


        prediction = predict_yield(

            crop_year,

            area,

            rainfall,

            fertilizer,

            pesticide

        )


        results = train_model()


        return render_template(

            "logistic_regression.html",

            active="logistic-regression",

            results=results,

            prediction=prediction,

            input_data={

                "crop_year":
                    crop_year,

                "area":
                    area,

                "rainfall":
                    rainfall,

                "fertilizer":
                    fertilizer,

                "pesticide":
                    pesticide

            },

            error=None

        )


    except Exception as e:

        return render_template(

            "logistic_regression.html",

            active="logistic-regression",

            results=None,

            prediction=None,

            input_data=None,

            error=(
                f"{type(e).__name__}: "
                f"{str(e)}"
            )

        )


# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

@app.route(
    "/download-logistic-yield"
)
def download_logistic_yield():

    prediction_file = os.path.join(

        BASE_DIR,

        "logistic_yield_predictions.csv"

    )


    if not os.path.exists(
        prediction_file
    ):

        return (
            "Prediction file has not "
            "been generated yet.",
            404
        )


    return send_file(

        prediction_file,

        as_attachment=True,

        download_name=
        "logistic_yield_predictions.csv"

    )
# ============================================================
# TREES
# ============================================================

@app.route("/trees")
def trees_page():

    error = None
    results = None

    try:

        results = run_decision_tree()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = (
            f"{type(e).__name__}: {str(e)}"
        )

    return render_template(

        "trees.html",

        active="trees",

        results=results,

        error=error

    )
# ============================================================
# DOWNLOAD TREE PREDICTIONS
# ============================================================

@app.route("/download-tree-predictions")
def download_tree_predictions():

    prediction_file = os.path.join(

        BASE_DIR,

        "tree_predictions.csv"

    )

    if not os.path.exists(
        prediction_file
    ):

        return (

            "Decision Tree predictions have not "
            "been generated yet. Run the model first.",

            404

        )

    return send_file(

        prediction_file,

        as_attachment=True,

        download_name="crop_yield_tree_predictions.csv"

    )
# ============================================================
# K-MEANS CLUSTERING
# ============================================================
@app.route("/kmeans", methods=["GET", "POST"])
def kmeans_page():

    error = None
    results = None

    method = "manual"
    manual_k = 3

    if request.method == "POST":

        method = request.form.get("method", "manual")

        try:
            manual_k = int(
                request.form.get("manual_k", 3)
            )
        except (ValueError, TypeError):
            manual_k = 3

        try:

            results = run_kmeans(
                method=method,
                manual_k_value=manual_k
            )

        except Exception as e:

            error = (
                f"{type(e).__name__}: {str(e)}"
            )

    return render_template(
        "kmeans.html",
        active="kmeans",
        results=results,
        error=error,
        selected_method=method,
        manual_k=manual_k
    )
@app.route("/download-kmeans-predictions")
def download_kmeans_predictions():

    import os

    file_path = os.path.join(
        BASE_DIR,
        "crop_kmeans_predictions.csv"
    )

    if not os.path.exists(file_path):
        return (
            "K-Means predictions file has not been generated yet.",
            404
        )

    return send_file(
        file_path,
        as_attachment=True,
        download_name="crop_kmeans_predictions.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
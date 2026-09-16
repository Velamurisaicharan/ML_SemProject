import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import (
    Flask,
    render_template,
    send_file
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


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        df = load_data()

        summary = get_data_summary(df)

    except Exception as e:

        error = str(e)

    return render_template(

        "index.html",

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
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
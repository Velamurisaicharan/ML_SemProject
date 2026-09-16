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
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
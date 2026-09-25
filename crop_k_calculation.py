import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# MANUAL K
# ============================================================

def manual_k():

    return 3


# ============================================================
# ELBOW METHOD
# ============================================================

def elbow_method(
    X,
    min_k=2,
    max_k=10
):

    wcss_values = []

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    # --------------------------------------------------------
    # Calculate WCSS for each K
    # --------------------------------------------------------

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        wcss_values.append(
            model.inertia_
        )

    # --------------------------------------------------------
    # Automatic elbow detection
    #
    # IMPORTANT:
    # Do NOT use np.cross() here.
    # New NumPy versions don't support the old
    # 2-dimensional vector behaviour.
    # --------------------------------------------------------

    points = np.column_stack(
        (
            np.array(k_values, dtype=float),
            np.array(wcss_values, dtype=float)
        )
    )

    first = points[0]

    last = points[-1]

    line_vector = last - first

    line_length = np.linalg.norm(
        line_vector
    )

    distances = []

    if line_length == 0:

        selected_k = k_values[0]

    else:

        # ----------------------------------------------------
        # Calculate perpendicular distance manually
        #
        # For two 2D vectors:
        #
        # |x1*y2 - y1*x2|
        #
        # ----------------------------------------------------

        for point in points:

            vector = point - first

            cross_value = abs(

                line_vector[0] * vector[1]
                -
                line_vector[1] * vector[0]

            )

            distance = (
                cross_value /
                line_length
            )

            distances.append(
                distance
            )

        selected_index = int(
            np.argmax(distances)
        )

        selected_k = (
            k_values[selected_index]
        )

    return {

        "k_values":
            k_values,

        "wcss":
            wcss_values,

        "distances":
            distances,

        "selected_k":
            selected_k

    }


# ============================================================
# SILHOUETTE METHOD
# ============================================================

def silhouette_method(
    X,
    min_k=2,
    max_k=10
):

    scores = []

    k_values = list(
        range(
            min_k,
            max_k + 1
        )
    )

    # --------------------------------------------------------
    # Calculate silhouette score
    # --------------------------------------------------------

    for k in k_values:

        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )

        clusters = model.fit_predict(
            X
        )

        # ----------------------------------------------------
        # Use sample for large datasets
        # ----------------------------------------------------

        sample_size = min(
            5000,
            len(X)
        )

        score = silhouette_score(

            X,

            clusters,

            sample_size=sample_size,

            random_state=42

        )

        scores.append(
            score
        )

    # --------------------------------------------------------
    # Select K having maximum silhouette score
    # --------------------------------------------------------

    best_index = int(
        np.argmax(scores)
    )

    selected_k = (
        k_values[best_index]
    )

    return {

        "k_values":
            k_values,

        "scores":
            scores,

        "selected_k":
            selected_k

    }


# ============================================================
# MAIN K CALCULATION
# ============================================================

def calculate_k(
    X,
    method="silhouette"
):

    # --------------------------------------------------------
    # MANUAL
    # --------------------------------------------------------

    if method == "manual":

        return {

            "method":
                "Manual K",

            "selected_k":
                manual_k(),

            "details":
                None

        }


    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    elif method == "elbow":

        result = elbow_method(
            X
        )

        return {

            "method":
                "Elbow Method",

            "selected_k":
                result["selected_k"],

            "details":
                result

        }


    # --------------------------------------------------------
    # SILHOUETTE
    # --------------------------------------------------------

    elif method == "silhouette":

        result = silhouette_method(
            X
        )

        return {

            "method":
                "Silhouette Score",

            "selected_k":
                result["selected_k"],

            "details":
                result

        }


    # --------------------------------------------------------
    # INVALID METHOD
    # --------------------------------------------------------

    else:

        raise ValueError(

            "Invalid K selection method. "
            "Use manual, elbow, or silhouette."

        )
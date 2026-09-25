from sklearn.metrics import silhouette_score


# ============================================================
# K-MEANS PERFORMANCE
# ============================================================

def calculate_performance(
    X,
    clusters,
    kmeans_model
):

    # --------------------------------------------------------
    # WCSS
    # --------------------------------------------------------

    wcss = (
        kmeans_model.inertia_
    )


    # --------------------------------------------------------
    # SILHOUETTE
    # --------------------------------------------------------

    sample_size = min(
        5000,
        len(X)
    )


    silhouette = silhouette_score(

        X,

        clusters,

        sample_size=sample_size,

        random_state=42

    )


    return {

        "wcss":
            float(wcss),

        "silhouette":
            float(silhouette)

    }
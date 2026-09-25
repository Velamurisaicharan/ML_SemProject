import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "crop_yield_preprocessed.csv"
)

CHART_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_crop_data():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            "Dataset not found: "
            + DATASET_PATH
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    # --------------------------------------------------------
    # Remove target column from clustering
    # --------------------------------------------------------

    target_names = [
        "Yield",
        "yield",
        "Crop_Yield",
        "Crop Yield",
        "Yield_tons",
        "Yield_Production"
    ]

    for col in target_names:

        if col in numeric_df.columns:

            numeric_df = numeric_df.drop(
                columns=[col]
            )

            break

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    numeric_df = numeric_df.fillna(0)

    if numeric_df.shape[1] < 2:

        raise ValueError(
            "At least two numerical features "
            "are required for K-Means visualization."
        )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        numeric_df
    )

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=numeric_df.columns
    )

    return numeric_df, X_scaled


# ============================================================
# CREATE CLUSTER GRAPH
# ============================================================

def create_cluster_graph(
    X_scaled,
    clusters,
    k,
    feature_x,
    feature_y
):

    plt.figure(
        figsize=(10, 6)
    )

    x_position = X_scaled.columns.get_loc(
        feature_x
    )

    y_position = X_scaled.columns.get_loc(
        feature_y
    )

    x_values = X_scaled.iloc[
        :,
        x_position
    ]

    y_values = X_scaled.iloc[
        :,
        y_position
    ]

    plt.scatter(
        x_values,
        y_values,
        c=clusters,
        cmap="viridis",
        s=35,
        alpha=0.75
    )

    plt.xlabel(
        feature_x
    )

    plt.ylabel(
        feature_y
    )

    plt.title(
        f"K-Means Clustering - K = {k}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    chart_path = os.path.join(
        CHART_DIR,
        "crop_kmeans_clusters.png"
    )

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return chart_path


# ============================================================
# ELBOW GRAPH
# ============================================================

def create_elbow_graph(
    X_scaled,
    max_k=10
):

    max_k = min(
        max_k,
        len(X_scaled) - 1
    )

    k_values = list(
        range(2, max_k + 1)
    )

    wcss = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(
            X_scaled
        )

        wcss.append(
            model.inertia_
        )

    # --------------------------------------------------------
    # Automatic elbow
    # --------------------------------------------------------

    points = np.column_stack(
        (
            np.array(
                k_values,
                dtype=float
            ),
            np.array(
                wcss,
                dtype=float
            )
        )
    )

    first = points[0]
    last = points[-1]

    line = last - first

    line_length = np.linalg.norm(
        line
    )

    distances = []

    if line_length == 0:

        best_k = k_values[0]

    else:

        for point in points:

            vector = point - first

            distance = abs(
                line[0] * vector[1]
                -
                line[1] * vector[0]
            ) / line_length

            distances.append(
                distance
            )

        best_k = k_values[
            int(
                np.argmax(
                    distances
                )
            )
        ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        k_values,
        wcss,
        marker="o",
        linewidth=2
    )

    plt.axvline(
        best_k,
        linestyle="--",
        linewidth=2
    )

    plt.scatter(
        [best_k],
        [
            wcss[
                k_values.index(best_k)
            ]
        ],
        s=100
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "WCSS / Inertia"
    )

    plt.title(
        f"Elbow Method - Selected K = {best_k}"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    chart_path = os.path.join(
        CHART_DIR,
        "crop_kmeans_elbow.png"
    )

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return best_k, k_values, wcss


# ============================================================
# SILHOUETTE GRAPH
# ============================================================

def create_silhouette_graph(
    X_scaled,
    max_k=10
):

    max_k = min(
        max_k,
        len(X_scaled) - 1
    )

    k_values = list(
        range(2, max_k + 1)
    )

    scores = []

    sample_size = min(
        5000,
        len(X_scaled)
    )

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        scores.append(
            score
        )

    best_index = int(
        np.argmax(scores)
    )

    best_k = k_values[
        best_index
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        k_values,
        scores,
        marker="o",
        linewidth=2
    )

    plt.axvline(
        best_k,
        linestyle="--",
        linewidth=2
    )

    plt.scatter(
        [best_k],
        [scores[best_index]],
        s=100
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        f"Silhouette Method - Selected K = {best_k}"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    chart_path = os.path.join(
        CHART_DIR,
        "crop_kmeans_silhouette.png"
    )

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return best_k, k_values, scores


# ============================================================
# RUN K-MEANS
# ============================================================

def run_kmeans(
    method="manual",
    manual_k_value=None
):

    df = load_crop_data()

    numeric_df, X_scaled = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Select features for visualization
    # --------------------------------------------------------

    feature_x = X_scaled.columns[0]

    feature_y = X_scaled.columns[1]

    elbow_data = None

    silhouette_data = None

    # --------------------------------------------------------
    # MANUAL
    # --------------------------------------------------------

    if method == "manual":

        if manual_k_value is None:

            raise ValueError(
                "Please enter a K value."
            )

        try:

            k = int(
                manual_k_value
            )

        except ValueError:

            raise ValueError(
                "K must be an integer."
            )

        if k < 2:

            raise ValueError(
                "K must be at least 2."
            )

        if k >= len(X_scaled):

            raise ValueError(
                "K must be smaller than "
                "the number of records."
            )

    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    elif method == "elbow":

        k, k_values, wcss = (
            create_elbow_graph(
                X_scaled
            )
        )

        elbow_data = {
            "k_values": k_values,
            "wcss": wcss
        }

    # --------------------------------------------------------
    # SILHOUETTE
    # --------------------------------------------------------

    elif method == "silhouette":

        k, k_values, scores = (
            create_silhouette_graph(
                X_scaled
            )
        )

        silhouette_data = {
            "k_values": k_values,
            "scores": scores
        }

    else:

        raise ValueError(
            "Invalid method."
        )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(
        X_scaled
    )

    # --------------------------------------------------------
    # Add clusters to original data
    # --------------------------------------------------------

    output_df = df.copy()

    output_df[
        "Cluster"
    ] = clusters

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    cluster_counts = (
        pd.Series(clusters)
        .value_counts()
        .sort_index()
    )

    cluster_info = []

    for cluster_id, count in cluster_counts.items():

        cluster_info.append({

            "cluster":
                int(cluster_id) + 1,

            "count":
                int(count)

        })

    # --------------------------------------------------------
    # Silhouette score for final model
    # --------------------------------------------------------

    if k > 1:

        final_score = silhouette_score(
            X_scaled,
            clusters,
            sample_size=min(
                5000,
                len(X_scaled)
            ),
            random_state=42
        )

    else:

        final_score = 0

    # --------------------------------------------------------
    # Create cluster graph
    # --------------------------------------------------------

    create_cluster_graph(
        X_scaled,
        clusters,
        k,
        feature_x,
        feature_y
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_path = os.path.join(
        BASE_DIR,
        "crop_kmeans_predictions.csv"
    )

    output_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    return {

        "method":
            method,

        "k":
            int(k),

        "rows":
            len(df),

        "features":
            X_scaled.shape[1],

        "feature_x":
            feature_x,

        "feature_y":
            feature_y,

        "silhouette_score":
            round(
                float(final_score),
                4
            ),

        "cluster_info":
            cluster_info,

        "elbow":
            elbow_data,

        "silhouette":
            silhouette_data,

        "cluster_chart":
            "charts/crop_kmeans_clusters.png",

        "elbow_chart":
            "charts/crop_kmeans_elbow.png",

        "silhouette_chart":
            "charts/crop_kmeans_silhouette.png",

        "output_file":
            "crop_kmeans_predictions.csv"

    }
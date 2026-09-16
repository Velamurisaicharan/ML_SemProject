import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data


sns.set(style="whitegrid")


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CHARTS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

print("Saving charts to:", CHARTS_DIR)


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(name):
    path = _chart_path(name)

    print("Saving:", path)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN EDA FUNCTION
# ============================================================

def run_eda():

    df = load_data()

    charts = []


    # ========================================================
    # CLEAN COLUMN VALUES
    # ========================================================

    # Remove extra spaces from categorical columns
    for col in ["Crop", "Season", "State"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("=" * 70)
    print("1. DATA LOADING")
    print("=" * 70)

    print("Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())


    # ========================================================
    # 2. BASIC INFORMATION
    # ========================================================

    print("=" * 70)
    print("2. BASIC INFORMATION")
    print("=" * 70)

    print("\nData Types:")
    print(df.dtypes)

    print("\nNumerical Description:")
    print(df.describe())

    print("\nCategorical Description:")
    print(df.describe(include="object"))


    info = {
        "dtypes": df.dtypes.astype(str).to_dict(),

        "describe_num":
            df.describe().to_dict(),

        "describe_cat":
            df.describe(include="object").to_dict()
    }


    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    print("=" * 70)
    print("3. MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing = df.isnull().sum()

    missing_pct = (
        missing / len(df)
    ) * 100

    print("\nMissing Values:")
    print(missing)

    print("\nMissing Percentage:")
    print(missing_pct)


    if missing.sum() > 0:

        plt.figure(figsize=(12, 6))

        sns.heatmap(
            df.isnull(),
            cbar=False,
            yticklabels=False
        )

        plt.title(
            "Missing Value Heatmap",
            fontsize=14
        )

        _save("missing_heatmap.png")

        charts.append("missing_heatmap.png")


        # Fill numerical missing values
        num_cols = df.select_dtypes(
            include="number"
        ).columns

        df[num_cols] = df[num_cols].fillna(
            df[num_cols].median()
        )


    # ========================================================
    # 4. DUPLICATE ROW ANALYSIS
    # ========================================================

    print("=" * 70)
    print("4. DUPLICATE ROW ANALYSIS")
    print("=" * 70)

    duplicate_count = int(
        df.duplicated().sum()
    )

    print(
        "Duplicate Rows:",
        duplicate_count
    )


    # ========================================================
    # 5. CROP DISTRIBUTION
    # ========================================================

    print("=" * 70)
    print("5. CROP DISTRIBUTION")
    print("=" * 70)

    if "Crop" in df.columns:

        crop_counts = (
            df["Crop"]
            .value_counts()
            .head(15)
        )

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=crop_counts.values,
            y=crop_counts.index
        )

        plt.title(
            "Top 15 Crops by Number of Records"
        )

        plt.xlabel("Number of Records")
        plt.ylabel("Crop")

        _save("crop_distribution.png")

        charts.append(
            "crop_distribution.png"
        )


    # ========================================================
    # 6. NUMERICAL FEATURE DISTRIBUTIONS
    # ========================================================

    print("=" * 70)
    print("6. NUMERICAL FEATURE DISTRIBUTIONS")
    print("=" * 70)

    num_features = [
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield"
    ]

    for col in num_features:

        if col in df.columns:

            plt.figure(figsize=(8, 5))

            sns.histplot(
                df[col],
                kde=True
            )

            plt.title(
                f"{col} Distribution"
            )

            plt.xlabel(col)
            plt.ylabel("Frequency")

            _save(
                f"{col.lower()}_distribution.png"
            )

            charts.append(
                f"{col.lower()}_distribution.png"
            )


    # ========================================================
    # 7. OUTLIER DETECTION
    # ========================================================

    print("=" * 70)
    print("7. OUTLIER DETECTION")
    print("=" * 70)

    for col in num_features:

        if col in df.columns:

            plt.figure(figsize=(8, 4))

            sns.boxplot(
                x=df[col]
            )

            plt.title(
                f"{col} Boxplot"
            )

            plt.xlabel(col)

            _save(
                f"{col.lower()}_boxplot.png"
            )

            charts.append(
                f"{col.lower()}_boxplot.png"
            )


    # ========================================================
    # 8. CORRELATION ANALYSIS
    # ========================================================

    print("=" * 70)
    print("8. CORRELATION ANALYSIS")
    print("=" * 70)

    corr = df.corr(
        numeric_only=True
    )

    print("\nCorrelation Matrix:")
    print(corr)


    plt.figure(
        figsize=(11, 8)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Agricultural Features Correlation Heatmap"
    )

    _save("correlation_heatmap.png")

    charts.append(
        "correlation_heatmap.png"
    )


    # ========================================================
    # 9. RELATIONSHIP ANALYSIS
    # ========================================================

    print("=" * 70)
    print("9. RELATIONSHIP ANALYSIS")
    print("=" * 70)


    # Area vs Production
    if (
        "Area" in df.columns
        and
        "Production" in df.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            x="Area",
            y="Production",
            data=df,
            alpha=0.5
        )

        plt.title(
            "Area vs Production"
        )

        _save(
            "area_vs_production.png"
        )

        charts.append(
            "area_vs_production.png"
        )


    # Rainfall vs Yield
    if (
        "Annual_Rainfall" in df.columns
        and
        "Yield" in df.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            x="Annual_Rainfall",
            y="Yield",
            data=df,
            alpha=0.5
        )

        plt.title(
            "Annual Rainfall vs Yield"
        )

        _save(
            "rainfall_vs_yield.png"
        )

        charts.append(
            "rainfall_vs_yield.png"
        )


    # Fertilizer vs Yield
    if (
        "Fertilizer" in df.columns
        and
        "Yield" in df.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            x="Fertilizer",
            y="Yield",
            data=df,
            alpha=0.5
        )

        plt.title(
            "Fertilizer vs Yield"
        )

        _save(
            "fertilizer_vs_yield.png"
        )

        charts.append(
            "fertilizer_vs_yield.png"
        )


    # ========================================================
    # 10. CATEGORICAL FEATURE ANALYSIS
    # ========================================================

    print("=" * 70)
    print("10. CATEGORICAL FEATURE ANALYSIS")
    print("=" * 70)


    categorical_cols = [
        "Crop",
        "Season",
        "State"
    ]


    for col in categorical_cols:

        if col in df.columns:

            counts = (
                df[col]
                .value_counts()
                .head(15)
            )

            plt.figure(
                figsize=(10, 6)
            )

            sns.barplot(
                x=counts.values,
                y=counts.index
            )

            plt.title(
                f"Top 15 {col} Categories"
            )

            plt.xlabel(
                "Number of Records"
            )

            plt.ylabel(col)

            _save(
                f"{col.lower()}_count.png"
            )

            charts.append(
                f"{col.lower()}_count.png"
            )


    # ========================================================
    # 11. SEASON ANALYSIS
    # ========================================================

    print("=" * 70)
    print("11. SEASON ANALYSIS")
    print("=" * 70)

    if (
        "Season" in df.columns
        and
        "Yield" in df.columns
    ):

        plt.figure(
            figsize=(10, 6)
        )

        sns.boxplot(
            x="Season",
            y="Yield",
            data=df
        )

        plt.xticks(
            rotation=30
        )

        plt.title(
            "Yield Distribution by Season"
        )

        _save(
            "season_vs_yield.png"
        )

        charts.append(
            "season_vs_yield.png"
        )


    # ========================================================
    # 12. STATE / REGION ANALYSIS
    # ========================================================

    print("=" * 70)
    print("12. STATE ANALYSIS")
    print("=" * 70)

    if (
        "State" in df.columns
        and
        "Production" in df.columns
    ):

        state_production = (
            df.groupby("State")["Production"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(15)
        )


        plt.figure(
            figsize=(12, 6)
        )

        sns.barplot(
            x=state_production.values,
            y=state_production.index
        )

        plt.title(
            "Top 15 States by Total Production"
        )

        plt.xlabel(
            "Total Production"
        )

        plt.ylabel(
            "State"
        )

        _save(
            "state_production.png"
        )

        charts.append(
            "state_production.png"
        )


    # ========================================================
    # 13. YEAR-WISE CROP TREND
    # ========================================================

    print("=" * 70)
    print("13. YEAR-WISE CROP TREND")
    print("=" * 70)

    if (
        "Crop_Year" in df.columns
        and
        "Production" in df.columns
    ):

        yearly_production = (
            df.groupby("Crop_Year")
            ["Production"]
            .sum()
        )


        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            yearly_production.index,
            yearly_production.values,
            marker="o"
        )

        plt.title(
            "Year-wise Agricultural Production"
        )

        plt.xlabel(
            "Crop Year"
        )

        plt.ylabel(
            "Total Production"
        )

        plt.xticks(
            rotation=45
        )

        _save(
            "yearly_production.png"
        )

        charts.append(
            "yearly_production.png"
        )


    # ========================================================
    # 14. YIELD ANALYSIS
    # ========================================================

    print("=" * 70)
    print("14. YIELD ANALYSIS")
    print("=" * 70)


    if "Yield" in df.columns:

        # Yield distribution
        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            df["Yield"],
            kde=True
        )

        plt.title(
            "Crop Yield Distribution"
        )

        plt.xlabel(
            "Yield"
        )

        plt.ylabel(
            "Frequency"
        )

        _save(
            "yield_distribution.png"
        )

        charts.append(
            "yield_distribution.png"
        )


        # Yield by season
        if "Season" in df.columns:

            plt.figure(
                figsize=(10, 6)
            )

            sns.boxplot(
                x="Season",
                y="Yield",
                data=df
            )

            plt.xticks(
                rotation=30
            )

            plt.title(
                "Yield by Season"
            )

            _save(
                "yield_by_season.png"
            )

            charts.append(
                "yield_by_season.png"
            )


        # Top states by average yield
        if "State" in df.columns:

            state_yield = (
                df.groupby("State")["Yield"]
                .mean()
                .sort_values(
                    ascending=False
                )
                .head(15)
            )


            plt.figure(
                figsize=(12, 6)
            )

            sns.barplot(
                x=state_yield.values,
                y=state_yield.index
            )

            plt.title(
                "Top 15 States by Average Crop Yield"
            )

            plt.xlabel(
                "Average Yield"
            )

            plt.ylabel(
                "State"
            )

            _save(
                "state_average_yield.png"
            )

            charts.append(
                "state_average_yield.png"
            )


    # ========================================================
    # 15. PAIRPLOT
    # ========================================================

    print("=" * 70)
    print("15. FEATURE RELATIONSHIP PAIRPLOT")
    print("=" * 70)


    pair_cols = [
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield"
    ]


    pair_cols = [
        col
        for col in pair_cols
        if col in df.columns
    ]


    if len(pair_cols) >= 2:

        # Use sample to prevent extremely large pairplot
        sample_df = df[pair_cols].sample(
            min(1500, len(df)),
            random_state=42
        )


        sns.pairplot(
            sample_df,
            diag_kind="hist"
        )


        plt.savefig(
            _chart_path(
                "pairplot.png"
            ),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close()

        charts.append(
            "pairplot.png"
        )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results = {

        "shape": df.shape,

        "duplicate_count":
            duplicate_count,

        "missing_count":
            missing.to_dict(),

        "missing_percent":
            missing_pct.to_dict(),

        "charts":
            charts,

        "info":
            info
    }


    print("=" * 70)
    print("EDA COMPLETED")
    print("=" * 70)

    print(
        "Total charts generated:",
        len(charts)
    )

    print(
        "Charts:",
        charts
    )


    return results
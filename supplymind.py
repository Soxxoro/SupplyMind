
# ============================================================
# SUPPLYMIND
# Supply Chain Disruption Prediction
#
# Phase 2:
# - Data Generation
# - Data Cleaning
# - Feature Engineering
# - EDA
# - Baseline Model
# - XGBoost
# - 5-Fold Cross Validation
# - Hyperparameter Tuning
# - Final Evaluation
# - Feature Importance
# - Model Saving
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# ============================================================
# 1. DATASET GENERATION
# ============================================================

def generate_dataset(
    n_rows=3000,
    random_state=RANDOM_STATE
):
    """
    Generate a realistic synthetic supply-chain dataset.
    """

    rng = np.random.default_rng(random_state)

    suppliers = [
        "Supplier_A",
        "Supplier_B",
        "Supplier_C",
        "Supplier_D",
        "Supplier_E",
    ]

    regions = [
        "North",
        "South",
        "East",
        "West",
        "Central",
    ]

    weather_conditions = [
        "Clear",
        "Rain",
        "Storm",
        "Flood",
        "Heatwave",
    ]

    shipment_modes = [
        "Road",
        "Rail",
        "Air",
        "Sea",
    ]

    product_categories = [
        "Electronics",
        "Pharma",
        "Food",
        "Automotive",
        "Textile",
    ]

    df = pd.DataFrame({

        "supplier": rng.choice(
            suppliers,
            n_rows
        ),

        "region": rng.choice(
            regions,
            n_rows
        ),

        "weather": rng.choice(
            weather_conditions,
            n_rows,
            p=[
                0.48,
                0.22,
                0.12,
                0.08,
                0.10,
            ]
        ),

        "shipment_mode": rng.choice(
            shipment_modes,
            n_rows,
            p=[
                0.45,
                0.20,
                0.15,
                0.20,
            ]
        ),

        "product_category": rng.choice(
            product_categories,
            n_rows
        ),

        "distance_km": np.clip(
            rng.normal(
                550,
                260,
                n_rows
            ),
            50,
            1800
        ).round(0),

        "supplier_reliability": np.clip(
            rng.normal(
                0.78,
                0.15,
                n_rows
            ),
            0.20,
            1.00
        ).round(3),

        "lead_time_days": np.clip(
            rng.normal(
                6,
                3,
                n_rows
            ),
            1,
            20
        ).round(1),

        "order_volume": np.clip(
            rng.normal(
                420,
                180,
                n_rows
            ),
            20,
            1200
        ).round(0),

        "inventory_level": np.clip(
            rng.normal(
                500,
                220,
                n_rows
            ),
            20,
            1500
        ).round(0),

        "route_risk": np.clip(
            rng.normal(
                0.35,
                0.20,
                n_rows
            ),
            0.01,
            1.00
        ).round(3),

        "traffic_index": np.clip(
            rng.normal(
                0.50,
                0.22,
                n_rows
            ),
            0.01,
            1.00
        ).round(3),
    })

    # ------------------------------------------
    # Business logic for disruption probability
    # ------------------------------------------

    weather_risk = {
        "Clear": 0.00,
        "Rain": 0.10,
        "Storm": 0.25,
        "Flood": 0.40,
        "Heatwave": 0.15,
    }

    mode_risk = {
        "Road": 0.08,
        "Rail": 0.04,
        "Air": 0.01,
        "Sea": 0.10,
    }

    weather_score = df["weather"].map(
        weather_risk
    )

    mode_score = df["shipment_mode"].map(
        mode_risk
    )

    risk_score = (
        0.28 * (
            1 - df["supplier_reliability"]
        )
        + 0.16 * df["route_risk"]
        + 0.14 * df["traffic_index"]
        + 0.10 * (
            df["lead_time_days"] / 20
        )
        + 0.08 * (
            df["distance_km"] / 1800
        )
        + 0.06 * (
            df["order_volume"] / 1200
        )
        + weather_score
        + mode_score
    )

    # Add realistic noise
    risk_score += rng.normal(
        0,
        0.05,
        n_rows
    )

    # Top 28% considered disrupted
    threshold = np.quantile(
        risk_score,
        0.72
    )

    df["disrupted"] = (
        risk_score > threshold
    ).astype(int)

    # ------------------------------------------
    # Add missing values
    # ------------------------------------------

    missing_1 = rng.choice(
        n_rows,
        70,
        replace=False
    )

    missing_2 = rng.choice(
        n_rows,
        50,
        replace=False
    )

    missing_3 = rng.choice(
        n_rows,
        40,
        replace=False
    )

    df.loc[
        missing_1,
        "supplier_reliability"
    ] = np.nan

    df.loc[
        missing_2,
        "traffic_index"
    ] = np.nan

    df.loc[
        missing_3,
        "lead_time_days"
    ] = np.nan

    return df


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    csv_path = os.path.join(
        DATA_DIR,
        "supply_chain_data.csv"
    )

    if os.path.exists(csv_path):

        print("\nExisting dataset found.")

        df = pd.read_csv(
            csv_path
        )

    else:

        print(
            "\nCreating supply-chain dataset..."
        )

        df = generate_dataset()

        df.to_csv(
            csv_path,
            index=False
        )

    return df


# ============================================================
# 3. DATA INSPECTION
# ============================================================

def inspect_data(df):

    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print(
        f"\nRows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )

    print("\nColumn names:")

    print(
        df.columns.tolist()
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    print("\nTarget distribution:")

    print(
        df["disrupted"]
        .value_counts()
    )

    print("\nTarget percentage:")

    print(
        (
            df["disrupted"]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


# ============================================================
# 4. DATA CLEANING
# ============================================================

def clean_data(df):

    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"\nDuplicate rows removed: "
        f"{before - after}"
    )

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(
            " ",
            "_"
        )
    )

    numeric_columns = [
        "distance_km",
        "supplier_reliability",
        "lead_time_days",
        "order_volume",
        "inventory_level",
        "route_risk",
        "traffic_index",
        "disrupted",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print(
        "\nData cleaning completed."
    )

    return df


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Inventory efficiency
    df[
        "inventory_to_order_ratio"
    ] = (
        df["inventory_level"]
        / (df["order_volume"] + 1)
    )

    # Long distance flag
    df["long_distance"] = (
        df["distance_km"] > 900
    ).astype(int)

    # High traffic flag
    df["high_traffic"] = (
        df["traffic_index"] > 0.70
    ).astype(int)

    # Supplier risk
    df["supplier_risk"] = (
        1 - df["supplier_reliability"]
    )

    # Long lead-time flag
    df["long_lead_time"] = (
        df["lead_time_days"] > 8
    ).astype(int)

    print("\nFeatures created:")

    print(
        [
            "inventory_to_order_ratio",
            "long_distance",
            "high_traffic",
            "supplier_risk",
            "long_lead_time",
        ]
    )

    return df


# ============================================================
# 6. EDA
# ============================================================

def perform_eda(df):

    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ------------------------------------------
    # Target Distribution
    # ------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    (
        df["disrupted"]
        .value_counts()
        .sort_index()
        .plot(
            kind="bar"
        )
    )

    plt.title(
        "Shipment Disruption Distribution"
    )

    plt.xlabel(
        "Disrupted "
        "(0 = No, 1 = Yes)"
    )

    plt.ylabel(
        "Number of Shipments"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "target_distribution.png"
        ),
        dpi=300
    )

    plt.show()

    # ------------------------------------------
    # Weather Analysis
    # ------------------------------------------

    weather_disruption = pd.crosstab(
        df["weather"],
        df["disrupted"],
        normalize="index"
    )

    weather_disruption.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Disruption Rate by Weather"
    )

    plt.xlabel(
        "Weather Condition"
    )

    plt.ylabel(
        "Proportion"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "weather_disruption.png"
        ),
        dpi=300
    )

    plt.show()

    # ------------------------------------------
    # Supplier Reliability
    # ------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    df.boxplot(
        column="supplier_reliability",
        by="disrupted"
    )

    plt.title(
        "Supplier Reliability vs Disruption"
    )

    plt.suptitle("")

    plt.xlabel(
        "Disrupted"
    )

    plt.ylabel(
        "Supplier Reliability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "supplier_reliability.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 7. PREPROCESSING
# ============================================================

def create_preprocessor():

    categorical_features = [
        "supplier",
        "region",
        "weather",
        "shipment_mode",
        "product_category",
    ]

    numeric_features = [
        "distance_km",
        "supplier_reliability",
        "lead_time_days",
        "order_volume",
        "inventory_level",
        "route_risk",
        "traffic_index",
        "inventory_to_order_ratio",
        "long_distance",
        "high_traffic",
        "supplier_risk",
        "long_lead_time",
    ]

    numeric_pipeline = Pipeline(
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
            ),
        ]
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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    return preprocessor


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

def prepare_data(df):

    X = df.drop(
        columns=["disrupted"]
    )

    y = df["disrupted"]

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 70)

    print(
        f"{name.upper()} - FINAL EVALUATION"
    )

    print("=" * 70)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Disruption",
                "Disruption"
            ],
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Disruption",
            "Disruption"
        ]
    )

    display.plot()

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.show()

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# 10. BASELINE MODEL
# ============================================================

def train_baseline(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\n" + "=" * 70)
    print(
        "TRAINING LOGISTIC REGRESSION BASELINE"
    )
    print("=" * 70)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        "Logistic Regression",
        model,
        X_test,
        y_test
    )

    return model, result


# ============================================================
# 11. INITIAL XGBOOST MODEL
# ============================================================

def train_initial_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\n" + "=" * 70)
    print(
        "TRAINING INITIAL XGBOOST"
    )
    print("=" * 70)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    result = evaluate_model(
        "Initial XGBoost",
        model,
        X_test,
        y_test
    )

    return model, result


# ============================================================
# 12. CROSS VALIDATION + HYPERPARAMETER TUNING
# ============================================================

def tune_xgboost(
    X_train,
    y_train,
    preprocessor
):

    print("\n" + "=" * 70)
    print(
        "CROSS VALIDATION + HYPERPARAMETER TUNING"
    )
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    )

    # Hyperparameter search space
    param_distributions = {

        "classifier__n_estimators": [
            100,
            200,
            300,
            400,
            500,
        ],

        "classifier__max_depth": [
            3,
            4,
            5,
            6,
            7,
            8,
        ],

        "classifier__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.08,
            0.10,
            0.15,
        ],

        "classifier__subsample": [
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "classifier__colsample_bytree": [
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "classifier__min_child_weight": [
            1,
            3,
            5,
            7,
        ],

        "classifier__gamma": [
            0,
            0.1,
            0.2,
            0.3,
        ],
    }

    # ------------------------------------------
    # 5-Fold Stratified Cross Validation
    # ------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # ------------------------------------------
    # Randomized Search
    # ------------------------------------------

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="f1",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    print(
        "\nStarting hyperparameter search..."
    )

    search.fit(
        X_train,
        y_train
    )

    print("\nBest CV F1 Score:")

    print(
        f"{search.best_score_:.4f}"
    )

    print(
        "\nBest Parameters:"
    )

    for key, value in search.best_params_.items():

        print(
            f"{key}: {value}"
        )

    return search


# ============================================================
# 13. SAVE CV RESULTS
# ============================================================

def save_cv_results(search):

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    columns_to_keep = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "params",
    ]

    cv_results = (
        cv_results[
            columns_to_keep
        ]
        .sort_values(
            "rank_test_score"
        )
    )

    path = os.path.join(
        OUTPUT_DIR,
        "cross_validation_results.csv"
    )

    cv_results.to_csv(
        path,
        index=False
    )

    print(
        f"\nCV results saved to: {path}"
    )


# ============================================================
# 14. PERMUTATION FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print(
        "PERMUTATION FEATURE IMPORTANCE"
    )
    print("=" * 70)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": (
            importance.importances_mean
        ),
    })

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .head(15)
    )

    print(
        "\nTop Features:"
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        importance_df[
            "feature"
        ][::-1],
        importance_df[
            "importance"
        ][::-1]
    )

    plt.title(
        "Top Features Influencing "
        "Disruption Prediction"
    )

    plt.xlabel(
        "Permutation Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.show()

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    return importance_df


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

def compare_models(results):

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    ]

    path = os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )

    comparison.to_csv(
        path,
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

def save_model(model):

    model_path = os.path.join(
        MODEL_DIR,
        "supplymind_xgboost.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nFinal model saved to:"
        f"\n{model_path}"
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "             SUPPLYMIND"
    )

    print(
        "     SUPPLY CHAIN DISRUPTION ML"
    )

    print("=" * 70)

    # ------------------------------------------
    # Step 1
    # ------------------------------------------

    df = load_data()

    # ------------------------------------------
    # Step 2
    # ------------------------------------------

    inspect_data(df)

    # ------------------------------------------
    # Step 3
    # ------------------------------------------

    df = clean_data(df)

    # ------------------------------------------
    # Step 4
    # ------------------------------------------

    df = feature_engineering(df)

    # ------------------------------------------
    # Step 5
    # ------------------------------------------

    perform_eda(df)

    # ------------------------------------------
    # Step 6
    # ------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    # ------------------------------------------
    # Step 7
    # ------------------------------------------

    preprocessor = create_preprocessor()

    # ------------------------------------------
    # Step 8
    # Baseline
    # ------------------------------------------

    (
        baseline_model,
        baseline_result
    ) = train_baseline(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # ------------------------------------------
    # Step 9
    # Initial XGBoost
    # ------------------------------------------

    (
        initial_xgb,
        initial_result
    ) = train_initial_xgboost(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # ------------------------------------------
    # Step 10
    # Hyperparameter Tuning
    # ------------------------------------------

    search = tune_xgboost(
        X_train,
        y_train,
        preprocessor
    )

    # ------------------------------------------
    # Step 11
    # Best Model Evaluation
    # ------------------------------------------

    best_model = search.best_estimator_

    best_result = evaluate_model(
        "Tuned XGBoost",
        best_model,
        X_test,
        y_test
    )

    # ------------------------------------------
    # Step 12
    # Save CV results
    # ------------------------------------------

    save_cv_results(
        search
    )

    # ------------------------------------------
    # Step 13
    # Feature Importance
    # ------------------------------------------

    calculate_feature_importance(
        best_model,
        X_test,
        y_test
    )

    # ------------------------------------------
    # Step 14
    # Compare all models
    # ------------------------------------------

    results = [
        baseline_result,
        initial_result,
        best_result,
    ]

    compare_models(
        results
    )

    # ------------------------------------------
    # Step 15
    # Save Model
    # ------------------------------------------

    save_model(
        best_model
    )

    # ------------------------------------------
    # Complete
    # ------------------------------------------

    print("\n" + "=" * 70)

    print(
        "SUPPLYMIND PHASE 2 COMPLETED"
    )

    print("=" * 70)

    print(
        "\nGenerated folders:"
    )

    print(
        "\ndata/"
    )

    print(
        "  └── supply_chain_data.csv"
    )

    print(
        "\noutputs/"
    )

    print(
        "  ├── target_distribution.png"
    )

    print(
        "  ├── weather_disruption.png"
    )

    print(
        "  ├── supplier_reliability.png"
    )

    print(
        "  ├── confusion_matrix.png"
    )

    print(
        "  ├── feature_importance.png"
    )

    print(
        "  ├── feature_importance.csv"
    )

    print(
        "  ├── cross_validation_results.csv"
    )

    print(
        "  └── model_comparison.csv"
    )

    print(
        "\nmodels/"
    )

    print(
        "  └── supplymind_xgboost.pkl"
    )

    print(
        "\nNext stage:"
    )

    print(
        "Build the SupplyMind prediction API "
        "and Streamlit dashboard."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# ZEPTO CAPSTONE PROJECT
# MODULE 2 - ANALYTICS PIPELINE
# PART B - MODELING PIPELINE
# STEP 1: CLASSIFICATION MODELS
# ============================================================

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)


# ============================================================
# STEP 2: CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("model_outputs", exist_ok=True)


# ============================================================
# STEP 3: LOAD DATASET
# ============================================================

print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

# Read the dataset saved during Part A
df = pd.read_csv("titanic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# STEP 4: SELECT FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: SELECTING FEATURES AND TARGET")
print("=" * 60)

# Target column
target = "survived"

# Select useful features
# Avoid target leakage and unnecessary derived columns
selected_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[selected_features].copy()
y = df[target].copy()

print("\nFeatures used:")
print(selected_features)

print("\nTarget:")
print(target)

print("\nTarget class distribution:")
print(y.value_counts())

print("\nTarget class percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# STEP 5: STRATIFIED TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: STRATIFIED TRAIN-TEST SPLIT")
print("=" * 60)

# Stratification keeps the survival class proportion
# approximately similar in train and test data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

print("""
Why stratification is used:
The target contains two classes: survived and did not survive.
Stratification maintains a similar class proportion in the
training and testing datasets, making model evaluation more
representative.
""")


# ============================================================
# STEP 6: DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: DEFINING PREPROCESSING")
print("=" * 60)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# ============================================================
# STEP 7: CREATE TRAIN-ONLY PREPROCESSING PIPELINE
# ============================================================

# Numerical preprocessing:
# 1. Fill missing values using median
# 2. Standardize numerical values

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# 1. Fill missing values using most frequent value
# 2. Convert categories into numerical columns

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


# Combine numerical and categorical preprocessing
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
        )
    ]
)

print("Preprocessing pipeline created successfully.")


# ============================================================
# STEP 8: DEFINE THREE CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DEFINING CLASSIFICATION MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

print("Three models created:")
for model_name in models:
    print("-", model_name)


# ============================================================
# STEP 9: TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: TRAINING AND EVALUATION")
print("=" * 60)

results = []
trained_pipelines = {}
predictions = {}
probabilities = {}


for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)
    print("-" * 60)

    # Complete pipeline:
    # preprocessing + model
    full_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Fit only on training data
    full_pipeline.fit(X_train, y_train)

    # Predict test data
    y_pred = full_pipeline.predict(X_test)

    # Predict probabilities for ROC curve
    y_probability = full_pipeline.predict_proba(
        X_test
    )[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    auc_score = roc_auc_score(
        y_test,
        y_probability
    )

    # Save results
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc_score
        }
    )

    # Save trained pipeline and predictions
    trained_pipelines[model_name] = full_pipeline
    predictions[model_name] = y_pred
    probabilities[model_name] = y_probability

    # Print metrics
    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("AUC:", round(auc_score, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# STEP 10: MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
print(comparison_df.round(4).to_string(index=False))

# Save comparison table
comparison_df.to_csv(
    "model_outputs/classification_comparison.csv",
    index=False
)

print(
    "\nComparison table saved as "
    "classification_comparison.csv"
)


# ============================================================
# STEP 11: CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for axis, (model_name, y_pred) in zip(
    axes,
    predictions.items()
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did Not Survive", "Survived"]
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False
    )

    axis.set_title(model_name)

plt.tight_layout()

plt.savefig(
    "model_outputs/confusion_matrices.png",
    dpi=300
)

plt.close()

print("Confusion matrices saved successfully.")


# ============================================================
# STEP 12: ROC CURVES AND AUC
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: ROC CURVES")
print("=" * 60)

plt.figure(figsize=(9, 6))

for model_name, y_probability in probabilities.items():

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    auc_score = roc_auc_score(
        y_test,
        y_probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )


# Random classification baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Baseline"
)

plt.title("ROC Curves for Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "model_outputs/roc_curves.png",
    dpi=300
)

plt.close()

print("ROC curves saved successfully.")


# ============================================================
# STEP 13: DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: DECISION TREE VISUALIZATION")
print("=" * 60)

# Retrieve trained Decision Tree pipeline
decision_tree_pipeline = trained_pipelines[
    "Decision Tree"
]

# Transform training data using fitted preprocessor
X_train_transformed = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .transform(X_train)
)

decision_tree_model = (
    decision_tree_pipeline
    .named_steps["model"]
)

# Get transformed feature names
feature_names = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=["Did Not Survive", "Survived"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Visualization")

plt.tight_layout()

plt.savefig(
    "model_outputs/decision_tree.png",
    dpi=300
)

plt.close()

print("Decision tree visualization saved successfully.")


# ============================================================
# STEP 14: SAVE BASELINE RANDOM FOREST PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: SAVING BASELINE PIPELINE")
print("=" * 60)

random_forest_pipeline = trained_pipelines[
    "Random Forest"
]

joblib.dump(
    random_forest_pipeline,
    "model_outputs/random_forest_baseline_pipeline.pkl"
)

print(
    "Baseline Random Forest pipeline saved successfully."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION MODELING COMPLETED")
print("=" * 60)

print("""
Completed:
1. Stratified train-test split
2. Train-only preprocessing
3. Logistic Regression
4. Decision Tree
5. Random Forest
6. Accuracy, Precision, Recall and F1
7. Confusion matrices
8. ROC curves and AUC
9. Model comparison table
10. Decision tree visualization
11. Baseline Random Forest pipeline saved
""")
# ============================================================
# STEP 12: CLASS IMBALANCE ANALYSIS
# BASELINE VS CLASS WEIGHT VS SMOTE
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: CLASS IMBALANCE ANALYSIS")
print("=" * 60)


# ============================================================
# STEP 12.1: CHECK CLASS BALANCE
# ============================================================

print("\nClass Distribution in Training Data:")
print(y_train.value_counts())

print("\nClass Distribution Percentage:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("""
Class Balance Interpretation:
The target variable contains two classes:
0 = Did not survive
1 = Survived

The class distribution is not perfectly equal.
Therefore, we compare the baseline model with
class-weight balancing and SMOTE.
""")


# ============================================================
# STEP 12.2: TRAIN BALANCED RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING BALANCED RANDOM FOREST")
print("=" * 60)


balanced_random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)


balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            balanced_random_forest
        )
    ]
)


# Fit only on training data
balanced_pipeline.fit(
    X_train,
    y_train
)

# Predict test data
balanced_predictions = balanced_pipeline.predict(
    X_test
)

balanced_probabilities = balanced_pipeline.predict_proba(
    X_test
)[:, 1]


# Calculate metrics
balanced_accuracy = accuracy_score(
    y_test,
    balanced_predictions
)

balanced_precision = precision_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_auc = roc_auc_score(
    y_test,
    balanced_probabilities
)


print("\nBalanced Random Forest Results:")
print("Accuracy:", round(balanced_accuracy, 4))
print("Precision:", round(balanced_precision, 4))
print("Recall:", round(balanced_recall, 4))
print("F1 Score:", round(balanced_f1, 4))
print("AUC:", round(balanced_auc, 4))


# ============================================================
# STEP 12.3: SMOTE PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST WITH SMOTE")
print("=" * 60)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# Create a separate preprocessing pipeline
# for the SMOTE experiment

smote_preprocessor = ColumnTransformer(
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
        )
    ]
)


smote_random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# SMOTE must be applied only to training data.
# The imbalanced-learn pipeline ensures that
# resampling does not happen on the test set.

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            smote_preprocessor
        ),
        (
            "smote",
            SMOTE(random_state=42)
        ),
        (
            "model",
            smote_random_forest
        )
    ]
)


# Fit SMOTE pipeline only on training data
smote_pipeline.fit(
    X_train,
    y_train
)


# Predict test data
smote_predictions = smote_pipeline.predict(
    X_test
)

smote_probabilities = smote_pipeline.predict_proba(
    X_test
)[:, 1]


# Calculate metrics
smote_accuracy = accuracy_score(
    y_test,
    smote_predictions
)

smote_precision = precision_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_predictions,
    zero_division=0
)

smote_auc = roc_auc_score(
    y_test,
    smote_probabilities
)


print("\nSMOTE Random Forest Results:")
print("Accuracy:", round(smote_accuracy, 4))
print("Precision:", round(smote_precision, 4))
print("Recall:", round(smote_recall, 4))
print("F1 Score:", round(smote_f1, 4))
print("AUC:", round(smote_auc, 4))


# ============================================================
# STEP 12.4: COMPARE THREE APPROACHES
# ============================================================

print("\n" + "=" * 60)
print("CLASS IMBALANCE COMPARISON")
print("=" * 60)


# Retrieve baseline Random Forest metrics
baseline_rf_result = comparison_df[
    comparison_df["Model"] == "Random Forest"
].iloc[0]


imbalance_results = pd.DataFrame(
    [
        {
            "Method": "Baseline Random Forest",
            "Accuracy": baseline_rf_result["Accuracy"],
            "Precision": baseline_rf_result["Precision"],
            "Recall": baseline_rf_result["Recall"],
            "F1 Score": baseline_rf_result["F1 Score"],
            "AUC": baseline_rf_result["AUC"]
        },
        {
            "Method": "Balanced Random Forest",
            "Accuracy": balanced_accuracy,
            "Precision": balanced_precision,
            "Recall": balanced_recall,
            "F1 Score": balanced_f1,
            "AUC": balanced_auc
        },
        {
            "Method": "SMOTE Random Forest",
            "Accuracy": smote_accuracy,
            "Precision": smote_precision,
            "Recall": smote_recall,
            "F1 Score": smote_f1,
            "AUC": smote_auc
        }
    ]
)


print("\nClass Imbalance Comparison Table:")
print(
    imbalance_results.round(4).to_string(index=False)
)


# Save comparison table
imbalance_results.to_csv(
    "model_outputs/class_imbalance_comparison.csv",
    index=False
)


print(
    "\nSaved: "
    "model_outputs/class_imbalance_comparison.csv"
)


# ============================================================
# STEP 12.5: CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("CLASS IMBALANCE CONFUSION MATRICES")
print("=" * 60)


imbalance_predictions = {
    "Baseline Random Forest": predictions["Random Forest"],
    "Balanced Random Forest": balanced_predictions,
    "SMOTE Random Forest": smote_predictions
}


fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4)
)


for axis, (method_name, method_predictions) in zip(
    axes,
    imbalance_predictions.items()
):

    cm = confusion_matrix(
        y_test,
        method_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did Not Survive", "Survived"]
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False
    )

    axis.set_title(method_name)


plt.tight_layout()

plt.savefig(
    "model_outputs/class_imbalance_confusion_matrices.png",
    dpi=300
)

plt.close()

print(
    "Class imbalance confusion matrices saved successfully."
)


# ============================================================
# STEP 12.6: AUTOMATIC METRIC COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("METRIC DIFFERENCES FROM BASELINE")
print("=" * 60)


baseline_recall = baseline_rf_result["Recall"]
baseline_f1 = baseline_rf_result["F1 Score"]
baseline_precision = baseline_rf_result["Precision"]


print("\nBalanced Random Forest Changes:")
print(
    "Precision change:",
    round(balanced_precision - baseline_precision, 4)
)

print(
    "Recall change:",
    round(balanced_recall - baseline_recall, 4)
)

print(
    "F1 Score change:",
    round(balanced_f1 - baseline_f1, 4)
)


print("\nSMOTE Random Forest Changes:")
print(
    "Precision change:",
    round(smote_precision - baseline_precision, 4)
)

print(
    "Recall change:",
    round(smote_recall - baseline_recall, 4)
)

print(
    "F1 Score change:",
    round(smote_f1 - baseline_f1, 4)
)


# ============================================================
# STEP 12.7: INTERPRETATION
# ============================================================

print("""
Interpretation:
The baseline, class-weight-balanced, and SMOTE models
are compared using precision, recall, and F1 score.

Class weighting changes the importance assigned to classes
during model training. SMOTE creates synthetic examples
for the minority class using only the training data.

The effect of each approach is evaluated on the unchanged
test set. The final approach should be selected based on
the project objective and the observed metric values.
""")


# ============================================================
# STEP 12 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("CLASS IMBALANCE ANALYSIS COMPLETED")
print("=" * 60)
# ============================================================
# STEP 13: RANDOM FOREST HYPERPARAMETER TUNING
# GRIDSEARCHCV AND OOB SCORE
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

from sklearn.model_selection import GridSearchCV


# ============================================================
# STEP 13.1: CREATE RANDOM FOREST WITH OOB SCORE
# ============================================================

tuning_random_forest = RandomForestClassifier(
    oob_score=True,
    bootstrap=True,
    random_state=42
)


# Complete pipeline:
# Preprocessing + Random Forest

tuning_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            tuning_random_forest
        )
    ]
)


# ============================================================
# STEP 13.2: DEFINE HYPERPARAMETER GRID
# ============================================================

parameter_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


print("\nHyperparameters being tested:")
print(parameter_grid)


# ============================================================
# STEP 13.3: CREATE GRIDSEARCHCV
# ============================================================

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# STEP 13.4: FIT GRIDSEARCHCV ON TRAINING DATA
# ============================================================

print("\nStarting GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 13.5: DISPLAY BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)
print("GRIDSEARCHCV RESULTS")
print("=" * 60)

print("\nBest Parameters:")

for parameter, value in grid_search.best_params_.items():
    print(parameter, ":", value)


print(
    "\nBest Cross-Validation F1 Score:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# STEP 13.6: EVALUATE BEST MODEL ON TEST DATA
# ============================================================

best_tuned_pipeline = grid_search.best_estimator_

tuned_predictions = best_tuned_pipeline.predict(
    X_test
)

tuned_probabilities = best_tuned_pipeline.predict_proba(
    X_test
)[:, 1]


tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_auc = roc_auc_score(
    y_test,
    tuned_probabilities
)


print("\nTuned Random Forest Test Results:")
print("Accuracy:", round(tuned_accuracy, 4))
print("Precision:", round(tuned_precision, 4))
print("Recall:", round(tuned_recall, 4))
print("F1 Score:", round(tuned_f1, 4))
print("AUC:", round(tuned_auc, 4))


# ============================================================
# STEP 13.7: GET OOB SCORE
# ============================================================

tuned_random_forest = (
    best_tuned_pipeline
    .named_steps["model"]
)

oob_score = tuned_random_forest.oob_score_

print(
    "\nOut-of-Bag (OOB) Score:",
    round(oob_score, 4)
)


# ============================================================
# STEP 13.8: SAVE TUNING RESULTS
# ============================================================

tuning_results = pd.DataFrame(
    [
        {
            "Model": "Tuned Random Forest",
            "Accuracy": tuned_accuracy,
            "Precision": tuned_precision,
            "Recall": tuned_recall,
            "F1 Score": tuned_f1,
            "AUC": tuned_auc,
            "Best CV F1": grid_search.best_score_,
            "OOB Score": oob_score
        }
    ]
)


print("\nTuning Results:")
print(tuning_results.round(4).to_string(index=False))


tuning_results.to_csv(
    "model_outputs/random_forest_tuning_results.csv",
    index=False
)


# ============================================================
# STEP 13.9: SAVE BEST TUNED PIPELINE
# ============================================================

joblib.dump(
    best_tuned_pipeline,
    "model_outputs/best_tuned_random_forest_pipeline.pkl"
)


print(
    "\nBest tuned Random Forest pipeline saved successfully."
)


# ============================================================
# STEP 13 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("GRIDSEARCHCV AND OOB ANALYSIS COMPLETED")
print("=" * 60)
# ============================================================
# STEP 14: FARE REGRESSION
# MULTIVARIATE LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: FARE REGRESSION")
print("=" * 60)


# ============================================================
# STEP 14.1: IMPORT REGRESSION LIBRARIES
# ============================================================

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 14.2: SELECT REGRESSION FEATURES AND TARGET
# ============================================================

print("\nSelecting regression features...")

# We predict fare.
# Fare must not be included as an input feature.

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

regression_target = "fare"


X_reg = df[regression_features].copy()
y_reg = df[regression_target].copy()


# Remove rows where the target fare is missing.
# This is applied only to rows with a missing target.

valid_rows = y_reg.notna()

X_reg = X_reg.loc[valid_rows]
y_reg = y_reg.loc[valid_rows]


print("\nRegression features:")
print(regression_features)

print("\nRegression target:")
print(regression_target)

print("\nRegression dataset shape:")
print(X_reg.shape)


# ============================================================
# STEP 14.3: TRAIN-TEST SPLIT
# ============================================================

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)


print("\nTraining data shape:", X_reg_train.shape)
print("Testing data shape:", X_reg_test.shape)


# ============================================================
# STEP 14.4: CREATE REGRESSION PREPROCESSOR
# ============================================================

regression_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]


regression_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


regression_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            regression_numeric_pipeline,
            regression_numeric_features
        ),
        (
            "categorical",
            regression_categorical_pipeline,
            regression_categorical_features
        )
    ]
)


# ============================================================
# STEP 14.5: CREATE REGRESSION PIPELINE
# ============================================================

regression_model = LinearRegression()


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "model",
            regression_model
        )
    ]
)


# ============================================================
# STEP 14.6: TRAIN REGRESSION MODEL
# ============================================================

print("\nTraining Linear Regression model...")

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


# Predict fare values
y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


print("Regression model trained successfully.")


# ============================================================
# STEP 14.7: CALCULATE REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


# Number of observations
n = len(y_reg_test)


# Number of transformed predictor columns
X_reg_test_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
)

p = X_reg_test_transformed.shape[1]


# Adjusted R-squared
if n - p - 1 > 0:

    adjusted_r2 = 1 - (
        (1 - r2) * (n - 1) / (n - p - 1)
    )

else:

    adjusted_r2 = np.nan


print("\n" + "=" * 60)
print("REGRESSION RESULTS")
print("=" * 60)

print("\nMean Absolute Error (MAE):", round(mae, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R-squared (R2):", round(r2, 4))
print("Adjusted R-squared:", round(adjusted_r2, 4))

print("\nNumber of observations:", n)
print("Number of predictors:", p)


# ============================================================
# STEP 14.8: RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)


# Residual = actual value - predicted value
residuals = y_reg_test - y_reg_pred


print("\nResidual mean:", round(residuals.mean(), 4))
print("Residual standard deviation:", round(residuals.std(), 4))


# Residual correlation with predicted values
residual_prediction_correlation = np.corrcoef(
    y_reg_pred,
    residuals
)[0, 1]


print(
    "Residual-prediction correlation:",
    round(residual_prediction_correlation, 4)
)


# Divide predictions into four groups
# to compare residual spread

residual_analysis_df = pd.DataFrame(
    {
        "Predicted Fare": y_reg_pred,
        "Residual": residuals
    }
)

residual_analysis_df["Prediction Group"] = pd.qcut(
    residual_analysis_df["Predicted Fare"],
    q=4,
    duplicates="drop"
)


residual_spread = (
    residual_analysis_df
    .groupby("Prediction Group", observed=True)["Residual"]
    .std()
)


print("\nResidual standard deviation by prediction group:")
print(residual_spread.round(4))


# ============================================================
# STEP 14.9: CREATE RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Plot: Fare Regression")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "model_outputs/fare_regression_residual_plot.png",
    dpi=300
)

plt.close()


print("\nResidual plot saved successfully.")


# ============================================================
# STEP 14.10: SAVE REGRESSION METRICS
# ============================================================

regression_results = pd.DataFrame(
    [
        {
            "Model": "Multivariate Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted R2": adjusted_r2
        }
    ]
)


print("\nRegression Metrics Table:")
print(
    regression_results.round(4).to_string(index=False)
)


regression_results.to_csv(
    "model_outputs/regression_metrics.csv",
    index=False
)


# ============================================================
# STEP 14.11: SAVE REGRESSION PIPELINE
# ============================================================

joblib.dump(
    regression_pipeline,
    "model_outputs/fare_regression_pipeline.pkl"
)


print("\nFare regression pipeline saved successfully.")


# ============================================================
# STEP 14 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("FARE REGRESSION COMPLETED")
print("=" * 60)
# ============================================================
# STEP 15: FINAL MODEL COMPARISON AND RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: FINAL MODEL COMPARISON")
print("=" * 60)


# ============================================================
# STEP 15.1: LOAD SAVED CLASSIFICATION AND REGRESSION RESULTS
# ============================================================

from pathlib import Path

output_directory = Path("model_outputs")

classification_file = (
    output_directory / "classification_comparison.csv"
)

regression_file = (
    output_directory / "regression_metrics.csv"
)


# Check whether classification results exist
if classification_file.exists():

    classification_results = pd.read_csv(
        classification_file
    )

    print("\nClassification Results:")
    print(
        classification_results.round(4).to_string(
            index=False
        )
    )

else:

    print(
        "\nClassification comparison file was not found."
    )

    classification_results = pd.DataFrame()


# Check whether regression results exist
if regression_file.exists():

    regression_results_final = pd.read_csv(
        regression_file
    )

    print("\nRegression Results:")
    print(
        regression_results_final.round(4).to_string(
            index=False
        )
    )

else:

    print(
        "\nRegression metrics file was not found."
    )

    regression_results_final = pd.DataFrame()


# ============================================================
# STEP 15.2: SAVE FINAL METRIC GROUPS
# ============================================================

if not classification_results.empty:

    classification_results.to_csv(
        output_directory / "final_classification_metrics.csv",
        index=False
    )

if not regression_results_final.empty:

    regression_results_final.to_csv(
        output_directory / "final_regression_metrics.csv",
        index=False
    )


print("\nFinal metric files saved successfully.")


# ============================================================
# STEP 15.3: GENERATE FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL RECOMMENDATION")
print("=" * 60)


if not classification_results.empty:

    # Select the model with the highest F1 score.
    best_f1_row = classification_results.loc[
        classification_results["F1 Score"].idxmax()
    ]

    best_f1_model = best_f1_row["Model"]
    best_f1_value = best_f1_row["F1 Score"]

    print(
        f"\nThe model with the highest F1 score is "
        f"{best_f1_model}."
    )

    print(
        f"Its F1 score is {best_f1_value:.4f}."
    )

    print(
        "\nThe classification models were evaluated "
        "using accuracy, precision, recall, F1 score, "
        "and AUC."
    )

    print(
        "F1 score provides a balance between precision "
        "and recall."
    )

    print(
        "The final classification model should be selected "
        "according to the project objective and evaluation "
        "metrics."
    )

else:

    print(
        "\nClassification results are unavailable."
    )


if not regression_results_final.empty:

    regression_row = regression_results_final.iloc[0]

    print("\nRegression model:")
    print(regression_row["Model"])

    print(
        f"MAE: {regression_row['MAE']:.4f}"
    )

    print(
        f"RMSE: {regression_row['RMSE']:.4f}"
    )

    print(
        f"R-squared: {regression_row['R2']:.4f}"
    )

    print(
        f"Adjusted R-squared: "
        f"{regression_row['Adjusted R2']:.4f}"
    )

    print(
        "\nThe regression model explains part of the "
        "variation in passenger fare."
    )

    print(
        "The residual analysis showed that the error "
        "spread increased for higher predicted fares, "
        "suggesting possible heteroscedasticity."
    )


# ============================================================
# STEP 15.4: SAVE FINAL REPORT
# ============================================================

report_lines = []

report_lines.append(
    "FINAL MODEL COMPARISON AND RECOMMENDATION"
)

report_lines.append("=" * 60)

if not classification_results.empty:

    best_f1_row = classification_results.loc[
        classification_results["F1 Score"].idxmax()
    ]

    report_lines.append(
        f"\nHighest F1 classification model: "
        f"{best_f1_row['Model']}"
    )

    report_lines.append(
        f"F1 Score: {best_f1_row['F1 Score']:.4f}"
    )

    report_lines.append(
        "\nClassification models were compared using "
        "accuracy, precision, recall, F1 score, and AUC."
    )

    report_lines.append(
        "F1 score was considered because it balances "
        "precision and recall."
    )

if not regression_results_final.empty:

    regression_row = regression_results_final.iloc[0]

    report_lines.append(
        "\nRegression model: "
        f"{regression_row['Model']}"
    )

    report_lines.append(
        f"MAE: {regression_row['MAE']:.4f}"
    )

    report_lines.append(
        f"RMSE: {regression_row['RMSE']:.4f}"
    )

    report_lines.append(
        f"R2: {regression_row['R2']:.4f}"
    )

    report_lines.append(
        f"Adjusted R2: "
        f"{regression_row['Adjusted R2']:.4f}"
    )

    report_lines.append(
        "\nThe residual analysis suggests possible "
        "heteroscedasticity because the residual spread "
        "increases for higher predicted fare values."
    )

report_lines.append(
    "\nThe final model selection should consider the "
    "evaluation metrics and the objective of the project."
)


final_report_path = (
    output_directory / "final_model_report.txt"
)

with open(final_report_path, "w") as report_file:

    report_file.write(
        "\n".join(report_lines)
    )


print(
    f"\nFinal report saved to: {final_report_path}"
)


# ============================================================
# STEP 15.5: RELOAD SAVED PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: PIPELINE RELOAD TEST")
print("=" * 60)


# Search for a tuned Random Forest pipeline
tuned_pipeline_files = list(
    output_directory.glob("*tuned*.pkl")
)


if len(tuned_pipeline_files) > 0:

    selected_pipeline_path = tuned_pipeline_files[0]

else:

    # Use the baseline pipeline if the tuned pipeline
    # is not found.
    baseline_pipeline_files = list(
        output_directory.glob("*random*forest*.pkl")
    )

    if len(baseline_pipeline_files) > 0:

        selected_pipeline_path = baseline_pipeline_files[0]

    else:

        selected_pipeline_path = None


if selected_pipeline_path is not None:

    print(
        "\nLoading pipeline from:"
    )

    print(selected_pipeline_path)

    loaded_pipeline = joblib.load(
        selected_pipeline_path
    )

    print(
        "Pipeline loaded successfully."
    )


    # ========================================================
    # STEP 15.6: CREATE RAW SAMPLE INPUT
    # ========================================================

    sample_input = pd.DataFrame(
        {
            "pclass": [1],
            "sex": ["female"],
            "age": [25.0],
            "sibsp": [0],
            "parch": [0],
            "fare": [50.0],
            "embarked": ["S"]
        }
    )


    print("\nRaw sample input:")
    print(sample_input)


    # Make a prediction
    sample_prediction = loaded_pipeline.predict(
        sample_input
    )


    print("\nSample prediction:")
    print(sample_prediction)


    # Predict probability if available
    if hasattr(
        loaded_pipeline,
        "predict_proba"
    ):

        sample_probability = (
            loaded_pipeline.predict_proba(
                sample_input
            )
        )

        print(
            "\nPrediction probabilities:"
        )

        print(sample_probability)


    print(
        "\nPipeline reload and prediction "
        "completed successfully."
    )

else:

    print(
        "\nNo saved classification pipeline "
        "was found in model_outputs."
    )


# ============================================================
# STEP 15 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON COMPLETED")
print("=" * 60)
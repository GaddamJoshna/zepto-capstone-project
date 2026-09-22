# ============================================================
# ZEPTO CAPSTONE PROJECT
# MODULE 2 - ANALYTICS PIPELINE
# PART A - PROFILING, CLEANING AND DATA STORY
# ============================================================

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 2: CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("eda_outputs", exist_ok=True)


# ============================================================
# STEP 3: LOAD DATASET ONLY ONCE
# ============================================================

print("=" * 60)
print("STEP 1: LOADING TITANIC DATASET")
print("=" * 60)

# Load Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

# Save offline fallback immediately after loading
df.to_csv("titanic.csv", index=False)

print("Dataset loaded successfully!")
print("Offline fallback saved as titanic.csv")


# ============================================================
# STEP 4: DATA PROFILING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: DATA PROFILING")
print("=" * 60)

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Missing value percentage
print("\nMissing Value Percentage:")
missing_percentage = df.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]

print(missing_percentage.round(2))


# ============================================================
# STEP 5: MISSING VALUE STRATEGY
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: MISSING VALUE HANDLING")
print("=" * 60)

df_clean = df.copy()

print("\nMissing Value Strategy:")

# Display exact percentages for affected columns
for column in missing_percentage.index:
    print(
        f"{column}: {missing_percentage[column]:.2f}% missing"
    )

# ------------------------------------------------------------
# Threshold Rule:
# Under 5%       -> Drop affected rows
# 5% to 30%      -> Impute
# High missing   -> Drop column or encode missing category
# ------------------------------------------------------------

# Age: approximately 19.87% missing
# Between 5% and 30% -> Median imputation
age_missing_rate = missing_percentage.get("age", 0)

if 5 <= age_missing_rate <= 30:
    df_clean["age"] = df_clean["age"].fillna(
        df_clean["age"].median()
    )
    print(
        f"\nAge: {age_missing_rate:.2f}% missing "
        "-> Median imputation applied."
    )

# Embarked: approximately 0.22% missing
# Under 5% -> Drop affected rows
embarked_missing_rate = missing_percentage.get("embarked", 0)

if 0 < embarked_missing_rate < 5:
    df_clean = df_clean.dropna(subset=["embarked"])
    print(
        f"Embarked: {embarked_missing_rate:.2f}% missing "
        "-> Rows with missing values dropped."
    )

# Embark Town: approximately 0.22% missing
# Under 5% -> Drop affected rows
embark_town_missing_rate = missing_percentage.get(
    "embark_town", 0
)

if 0 < embark_town_missing_rate < 5:
    df_clean = df_clean.dropna(subset=["embark_town"])
    print(
        f"Embark Town: {embark_town_missing_rate:.2f}% missing "
        "-> Rows with missing values dropped."
    )

# Deck: approximately 77% missing
# High missingness -> Drop column
deck_missing_rate = missing_percentage.get("deck", 0)

if deck_missing_rate > 30:
    df_clean = df_clean.drop(columns=["deck"])
    print(
        f"Deck: {deck_missing_rate:.2f}% missing "
        "-> Column dropped due to high missingness."
    )

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

# Save cleaned dataset
df_clean.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as cleaned_titanic.csv")


# ============================================================
# STEP 6: UNIVARIATE ANALYSIS
# AGE AND FARE HISTOGRAMS + BOX PLOTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: UNIVARIATE ANALYSIS")
print("=" * 60)

sns.set_style("whitegrid")

# Create Age histogram and box plot

   # Age histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    data=df_clean,
    x="age",
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")

sns.boxplot(
    data=df_clean,
    x="age",
    ax=axes[1]
)

axes[1].set_title("Age Box Plot")
axes[1].set_xlabel("Age")

plt.tight_layout()
plt.savefig("eda_outputs/age_analysis.png", dpi=300)
plt.close()
# Fare histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    data=df_clean,
    x="fare",
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Fare Distribution")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Frequency")

sns.boxplot(
    data=df_clean,
    x="fare",
    ax=axes[1]
)

axes[1].set_title("Fare Box Plot")
axes[1].set_xlabel("Fare")

plt.tight_layout()
plt.savefig("eda_outputs/fare_analysis.png", dpi=300)
plt.close()

# ============================================================
# STEP 7: IQR OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: IQR OUTLIER DETECTION")
print("=" * 60)


def count_iqr_outliers(data, column):
    """
    Count outliers using the IQR rule.
    Outliers are values outside:
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    """

    values = data[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) |
        (values > upper_bound)
    ]

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    }


age_outliers = count_iqr_outliers(df_clean, "age")
fare_outliers = count_iqr_outliers(df_clean, "fare")

print("\nAge IQR Analysis:")
for key, value in age_outliers.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")

print("\nFare IQR Analysis:")
for key, value in fare_outliers.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float)
          else f"{key}: {value}")


# ============================================================
# STEP 8: FARE STATISTICS AND SKEWNESS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: FARE STATISTICS AND SKEWNESS")
print("=" * 60)

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode().iloc[0]
fare_skewness = df_clean["fare"].skew()

print(f"\nFare Mean: {fare_mean:.4f}")
print(f"Fare Median: {fare_median:.4f}")
print(f"Fare Mode: {fare_mode:.4f}")
print(f"Fare Skewness: {fare_skewness:.4f}")

if fare_mean > fare_median > fare_mode:
    skewness_conclusion = (
        "Fare is right-skewed based on the "
        "mean > median > mode ordering."
    )
elif fare_mean < fare_median < fare_mode:
    skewness_conclusion = (
        "Fare is left-skewed based on the "
        "mean < median < mode ordering."
    )
else:
    skewness_conclusion = (
        "The mean, median, and mode do not follow "
        "a strict ordering. Skewness is assessed "
        "using the skewness statistic as well."
    )

print("\nFare Distribution Interpretation:")
print(skewness_conclusion)


# ============================================================
# STEP 11: MULTIVARIATE DATA STORY
# FOUR DISTINCT CHARTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: MULTIVARIATE DATA STORY")
print("=" * 60)


# ------------------------------------------------------------
# CHART 1: SURVIVAL RATE BY SEX
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="sex",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_sex.png",
    dpi=300
)


plt.close("all")

print("Chart 1 saved: Survival by Sex")

print("""
Interpretation:
Survival rates differ between male and female passengers.
The chart shows gender-based differences in survival outcomes.
These results describe patterns in the dataset and do not
establish that gender alone caused survival.
""")


# ------------------------------------------------------------
# CHART 2: SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_pclass.png",
    dpi=300
)

plt.close("all")

print("Chart 2 saved: Survival by Passenger Class")

print("""
Interpretation:
Survival rates vary across passenger classes.
The chart compares survival outcomes for first, second,
and third class passengers. Passenger class is associated
with survival in this dataset, but the chart alone cannot
establish a causal relationship.
""")


# ------------------------------------------------------------
# CHART 3: SURVIVAL BY SEX AND PCLASS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_sex_pclass.png",
    dpi=300
)

plt.close("all")

print("Chart 3 saved: Survival by Sex and Passenger Class")

print("""
Interpretation:
This chart compares survival rates across passenger classes
for male and female passengers. It shows how the relationship
between gender and survival varies across passenger classes.
The combined analysis provides more detail than studying
each variable separately.
""")


# ------------------------------------------------------------
# CHART 4: AGE, FARE AND SURVIVAL
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age, Fare and Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig(
    "eda_outputs/age_fare_survival.png",
    dpi=300
)

plt.close("all")

print("Chart 4 saved: Age, Fare and Survival")

print("""
Interpretation:
This scatter plot explores the relationship between age,
fare, and survival status. Colors represent survival outcomes,
while marker styles represent gender. The chart helps identify
patterns and overlapping groups but does not establish causal
relationships between the variables.
""")

print("All four multivariate charts saved successfully!")
    
# ============================================================
# STEP 11: MULTIVARIATE DATA STORY
# FOUR DISTINCT CHARTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: MULTIVARIATE DATA STORY")
print("=" * 60)


# ------------------------------------------------------------
# CHART 1: SURVIVAL RATE BY SEX
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="sex",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Gender")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_sex.png",
    dpi=300
)
plt.close()

print("""
Interpretation - Survival by Gender:
Survival rates differ between male and female passengers.
The chart helps identify gender-based differences in survival
outcomes. These differences describe patterns in this dataset
and do not establish that gender alone caused survival.
""")


# ------------------------------------------------------------
# CHART 2: SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    errorbar=None
)

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_pclass.png",
    dpi=300
)

plt.close()

print("""
Interpretation - Survival by Passenger Class:
Survival rates vary across passenger classes.
The chart shows differences in survival outcomes between
first, second, and third class passengers. Passenger class
is associated with survival in this dataset, although the
chart alone cannot establish causation.
""")


# ------------------------------------------------------------
# CHART 3: SURVIVAL BY SEX AND PCLASS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title("Survival Rate by Gender and Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")

plt.tight_layout()
plt.savefig(
    "eda_outputs/survival_by_sex_pclass.png",
    dpi=300
)

plt.close()

print("""
Interpretation - Gender and Passenger Class:
This chart compares survival rates across passenger classes
for male and female passengers. It shows how the relationship
between gender and survival varies across passenger classes.
The combined breakdown provides more detail than either
variable considered separately.
""")


# ------------------------------------------------------------
# CHART 4: AGE, FARE AND SURVIVAL
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age, Fare and Survival")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig(
    "eda_outputs/age_fare_survival.png",
    dpi=300
)
plt.show(block=False)
plt.pause(3)

plt.close()

print("""
Interpretation - Age, Fare and Survival:
This scatter plot explores the relationship between age,
fare, and survival status. The colors represent survival
outcomes, while marker styles represent gender. The chart
helps identify patterns and overlapping groups, but it does
not establish causal relationships between the variables.
""")


# ============================================================
# STEP 12: STANDARDIZATION OF AGE AND FARE
# EDA-ONLY SANITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: STANDARDIZATION CHECK")
print("=" * 60)

standardization_columns = ["age", "fare"]

# Show before standardization
print("\nBefore Standardization:")

before_standardization = df_clean[
    standardization_columns
].agg(["mean", "std"])

print(before_standardization.round(4))

# Apply StandardScaler to age and fare
scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    df_clean[standardization_columns]
)

df_standardized = df_clean.copy()

df_standardized[
    ["age_standardized", "fare_standardized"]
] = standardized_values

# Show after standardization
print("\nAfter Standardization:")

after_standardization = df_standardized[
    ["age_standardized", "fare_standardized"]
].agg(["mean", "std"])

print(after_standardization.round(4))

print("""
Standardization Interpretation:
Age and fare were standardized using the z-score method.
The transformed columns have approximately zero mean and
unit population standard deviation. Pandas uses sample standard
deviation by default, so its displayed standard deviation
may be slightly different from 1. The transformation is
used only for this EDA sanity check and is not passed into
the later modeling pipeline.
""")


# ============================================================
# STEP 13: SAVE STANDARDIZED EDA DATA
# ============================================================

df_standardized.to_csv(
    "standardized_titanic_eda.csv",
    index=False
)

print("\nStandardized EDA dataset saved successfully!")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PART A - EDA COMPLETED")
print("=" * 60)

print("""
Completed:
1. Dataset loading and profiling
2. Missing value percentage analysis
3. Missing value handling
4. Age and Fare histograms
5. Age and Fare box plots
6. IQR outlier detection
7. Fare statistics and skewness
8. Survival rate analysis
9. Correlation matrix and heatmap
10. Two strongest correlation pairs
11. Four multivariate charts
12. Standardization check
13. Saved EDA output files
""")

print("Part A execution completed successfully!")
plt.show()
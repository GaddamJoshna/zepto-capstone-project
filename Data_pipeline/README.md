# Module 1 — Data Pipeline

## Purpose
Scrape → Clean → Fixed-rate currency conversion → Normalized SQLite → SQL + pandas queries.

## Fixed conversion rate (required)
**1 GBP = 105.50 INR**  
This is the project-defined constant.

## How to run
```bash
cd data_pipeline
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python scrape_and_load.py
Zepto Capstone Project – Titanic Analytics and Machine Learning

1. Project Overview

This project is developed as part of the Zepto Capstone Project. It
demonstrates a complete analytics and machine learning workflow using
the Titanic dataset.

The project is divided into two major parts:

• Part A – Exploratory Data Analysis (EDA)
• Part B – Machine Learning and Model Evaluation

The workflow includes data loading, data profiling, missing-value
analysis, data cleaning, visualization, classification, class-imbalance
handling, hyperparameter tuning, regression analysis, model saving, and
pipeline reloading.

────────

2. Project Objectives

The main objectives of this project are:

1. Load and inspect the Titanic dataset.
2. Analyze missing values and handle them using suitable strategies.
3. Perform univariate, bivariate, and multivariate analysis.
4. Study the relationship between passenger characteristics and
survival.
5. Build and compare classification models.
6. Analyze the effect of class imbalance.
7. Tune the Random Forest model using GridSearchCV.
8. Predict passenger fare using multivariate linear regression.
9. Evaluate models using appropriate performance metrics.
10. Save and reload a complete machine learning pipeline.

────────

3. Project Structure

Data_pipeline/
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── cleaned_titanic.csv
│   ├── standardized_titanic_eda.csv
│   ├── random_forest_model.pkl
│   │
│   ├── eda_outputs/
│   │   ├── age_analysis.png
│   │   ├── fare_analysis.png
│   │   ├── age_fare_survival.png
│   │   ├── correlation_heatmap.png
│   │   ├── survival_by_sex.png
│   │   ├── survival_by_pclass.png
│   │   └── survival_by_sex_pclass.png
│   │
│   └── model_outputs/
│       ├── classification_comparison.csv
│       ├── class_imbalance_comparison.csv
│       ├── confusion matrices
│       ├── ROC curves
│       ├── decision tree visualization
│       ├── tuned Random Forest pipeline
│       ├── fare regression pipeline
│       └── final_model_report.txt
│
├── requirements.txt
├── scrape_and_load.py
└── README.md

────────

4. Technologies and Libraries Used

• Python
• Pandas
• NumPy
• Matplotlib
• Seaborn
• Scikit-learn
• Imbalanced-learn
• Joblib

Main techniques used

• Data cleaning
• Missing-value handling
• Exploratory Data Analysis
• Boolean masking
• Correlation analysis
• Feature preprocessing
• Stratified train-test split
• Logistic Regression
• Decision Tree Classifier
• Random Forest Classifier
• GridSearchCV
• SMOTE
• Class-weight balancing
• Linear Regression
• Pipeline serialization using Joblib

────────

Part A – Exploratory Data Analysis

5. Dataset Loading

The Titanic dataset is loaded using Seaborn:

sns.load_dataset("titanic")

The original dataset is saved as:

analytics/titanic.csv

The same dataset is used for the remaining analysis and modeling
workflow.

────────

6. Dataset Profiling

The following profiling operations are performed:

• Dataset shape
• Data types
• Dataset information using df.info()
• Descriptive statistics using df.describe()
• Missing-value counts
• Missing-value percentages

The profiling step helps identify:

• Numerical columns
• Categorical columns
• Missing values
• Possible outliers
• The general distribution of the dataset

────────

7. Missing-Value Handling

Missing values are analyzed using their percentage of the total number
of records.

The following strategies are used:

────────

Column                  Handling Strategy       Reason

────────

age                   Median imputation       The column contains a
moderate percentage of
missing values. The
median reduces the
effect of outliers.

embarked              Mode-based handling or  The missing percentage
row removal according   is very small.
to the EDA rule

embark_town           Mode-based handling or  The missing percentage
removal after checking  is very small.
the related column

deck                  Column removal          The column contains a very high percentage of missing values, making reliable imputation difficult.

The cleaned dataset is saved as:

analytics/cleaned_titanic.csv

The missing-value strategy is selected according to the percentage of
missing values and the usefulness of the feature.

────────

8. Univariate Analysis

Univariate analysis is performed for the age and fare columns.

The following visualizations are created:

• Age histogram
• Age box plot
• Fare histogram
• Fare box plot

Outlier Detection

The Interquartile Range (IQR) method is used to identify possible
outliers.

The formula is:

IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Values below the lower bound or above the upper bound are considered
possible outliers.

Fare Distribution

The fare statistics include:

• Mean
• Median
• Mode
• Skewness

The observed ordering is:

Mean > Median > Mode

This indicates that the fare distribution is right-skewed. A small
number of passengers paid considerably higher fares than most
passengers.

────────

9. Bivariate Analysis

Survival rates are calculated for:

1. Passenger sex
2. Passenger class
3. Sex and passenger class together

Survival by Sex

The survival rate is higher for female passengers than for male
passengers in the dataset.

Survival by Passenger Class

Survival rates differ across passenger classes. First-class passengers
have a higher survival rate than second-class and third-class passengers
in this dataset.

Survival by Sex and Passenger Class

The combined analysis shows differences between male and female
passengers within each passenger class.

For example:

• Female first-class passengers have a high survival rate.
• Male third-class passengers have a much lower survival rate.

These results describe patterns in the dataset. They do not
independently establish causation.

────────

10. Correlation Analysis

A six-column correlation matrix is created using:

survived
pclass
age
sibsp
parch
fare

The correlation heatmap is saved in the EDA output directory.

Strongest Correlation Pairs

The two strongest absolute off-diagonal correlations identified are:

1. pclass and fare
  • Correlation: approximately -0.5482
  • Interpretation: Passenger class and fare have a moderate
negative relationship. Lower class numbers represent higher
passenger classes, which generally have higher fares.
2. sibsp and parch
  • Correlation: approximately 0.4145
  • Interpretation: Passengers travelling with siblings or spouses
were also more likely to travel with parents or children.

Correlation indicates association and does not prove causation.

────────

11. Multivariate Analysis

Four multivariate charts are created:

Chart 1: Survival by Sex

This chart compares survival outcomes between male and female
passengers.

Interpretation: Survival rates differ between the two sex
categories. The chart highlights a gender-based pattern in the dataset
but does not prove that sex alone caused the outcome.

Chart 2: Survival by Passenger Class

This chart compares survival rates across first, second, and third
class.

Interpretation: Survival outcomes vary across passenger classes.
First-class passengers show a higher survival rate than third-class
passengers in the dataset. Other factors may also be involved.

Chart 3: Survival by Sex and Passenger Class

This chart combines sex and passenger class.

Interpretation: The relationship between sex and survival changes
across passenger classes. Examining both variables together provides
more detail than studying either variable separately.

Chart 4: Age, Fare, and Survival

This chart studies age, fare, and survival together, with passenger sex
represented visually.

Interpretation: The chart displays patterns and overlapping groups
among age, fare, and survival. It helps explore relationships between
multiple variables, but it cannot independently establish causal
relationships.

────────

12. EDA Standardization Check

The age and fare columns are standardized using the Z-score method:

z = (x - mean) / standard deviation

After standardization:

• Mean is approximately 0
• Standard deviation is approximately 1

The standardized EDA dataset is saved as:

analytics/standardized_titanic_eda.csv

This standardization is used only for the EDA check. The final modeling
workflow uses preprocessing inside the machine learning pipeline to
avoid data leakage.

────────

Part B – Machine Learning

13. Classification Problem

The classification task predicts whether a passenger survived.

Target Column

survived

Selected Features

The classification pipeline uses passenger-related features such as:

• pclass
• sex
• age
• sibsp
• parch
• fare
• embarked

Columns that could cause target leakage or duplicate information are
excluded.

────────

14. Train-Test Split

A stratified train-test split is used.

The dataset is divided into:

• Training data: approximately 80%
• Testing data: approximately 20%

Stratification is used to preserve a similar distribution of the target
classes in both training and testing data.

This is important because the number of survivors and non-survivors is
not exactly equal.

────────

15. Train-Only Preprocessing

The preprocessing stage is fitted only on the training data.

The pipeline includes:

• Numerical missing-value imputation
• Categorical missing-value imputation
• One-hot encoding for categorical features
• Feature scaling where required
• Model training

Using a pipeline helps prevent data leakage from the test dataset into
the training process.

────────

16. Classification Models

Three classification algorithms are trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models are evaluated on the same train-test split.

Evaluation Metrics

The following metrics are calculated:

• Accuracy
• Precision
• Recall
• F1 score
• ROC-AUC
• Confusion matrix
• ROC curve

────────

17. Classification Results

The classification comparison produced the following results:

Model                   Accuracy   Precision   Recall   F1 Score      AUC

────────

Logistic Regression       0.8045      0.7931   0.6667     0.7244   0.8435
Decision Tree             0.7654      0.7547   0.5797     0.6557   0.7971
Random Forest             0.8101      0.7692   0.7246     0.7463   0.8300

The Random Forest model produced the highest F1 score among the three
evaluated classification models.

The F1 score is useful because it balances precision and recall.

────────

18. Class Imbalance Analysis

The following approaches are compared:

1. Baseline Random Forest
2. Random Forest with class_weight="balanced"
3. Random Forest with SMOTE applied to the training data

The test dataset remains unchanged for a fair comparison.

Class Imbalance Results

Method                     Accuracy   Precision   Recall   F1 Score      AUC

────────

Baseline Random Forest       0.8101      0.7692   0.7246     0.7463   0.8300
Balanced Random Forest       0.7989      0.7463   0.7246     0.7353   0.8326
SMOTE Random Forest          0.7933      0.7286   0.7391     0.7338   0.8345

The balanced and SMOTE approaches changed the precision-recall
trade-off.

SMOTE slightly increased recall compared with the baseline model, while
the baseline model produced a higher F1 score in the recorded results.

The appropriate approach depends on the project objective. If
identifying more positive cases is more important, recall may receive
greater importance.

────────

19. Random Forest Hyperparameter Tuning

GridSearchCV is used to tune the Random Forest classifier.

Hyperparameters Tested

• n_estimators: [100, 200]
• max_depth: [None, 5, 10]
• max_features: ["sqrt", "log2"]

The search uses five-fold cross-validation.

Best Parameters

The recorded best parameters were:

max_depth: 10
max_features: log2
n_estimators: 100

Tuning Results

Best Cross-Validation F1 Score: 0.7485
Test Accuracy: 0.8101
Test Precision: 0.7966
Test Recall: 0.6812
Test F1 Score: 0.7334
Test AUC: 0.8379
OOB Score: 0.8202

The Out-of-Bag score is calculated because the Random Forest model uses
bootstrapped samples and oob_score=True.

────────

20. Regression Problem

A multivariate linear regression model is used to predict passenger
fare.

Regression Target

fare

Predictor Features

The model uses available passenger information such as:

• Passenger class
• Sex
• Age
• Siblings or spouses aboard
• Parents or children aboard
• Embarked port

The fare column is excluded from the input features because it is the
target variable.

────────

21. Regression Results

The recorded regression results are:

Metric                               Value

────────

Mean Absolute Error (MAE)          20.8094
Root Mean Squared Error (RMSE)     30.4731
R-squared                           0.3999
Adjusted R-squared                  0.3753

Interpretation

The regression model explains part of the variation in passenger fare,
but a considerable amount of variation remains unexplained.

The RMSE is higher than the MAE because larger prediction errors receive
more weight in the RMSE calculation.

The residual analysis showed that the error spread increased for higher
predicted fares, suggesting possible heteroscedasticity.

────────

22. Model Saving and Pipeline Reloading

The complete machine learning pipelines are saved using Joblib.

Example:

joblib.dump(full_pipeline, "best_tuned_random_forest_pipeline.pkl")

The saved pipeline is then reloaded:

loaded_pipeline = joblib.load(
    "best_tuned_random_forest_pipeline.pkl"
)

A raw sample input is passed to the reloaded pipeline to confirm that
the saved preprocessing and model components work together.

The pipeline reload and prediction test completed successfully.

────────

23. Final Findings

The project produced the following findings:

1. Passenger sex and passenger class show noticeable differences in
survival rates.
2. Passenger class and fare have a moderate negative correlation.
3. Random Forest produced the highest recorded F1 score among the
initial classification models.
4. Class-weight balancing and SMOTE changed the precision and recall
trade-off.
5. GridSearchCV identified a tuned Random Forest configuration with
max_depth=10, max_features="log2", and n_estimators=100.
6. The fare regression model explained part of the fare variation but
showed possible heteroscedasticity.
7. Saving the complete preprocessing and modeling pipeline allows the
model to be reused on raw input data.

────────

24. How to Run the Project

Step 1: Open the project folder

Open the Data_pipeline folder in VS Code.

Step 2: Activate the virtual environment

source .venv/bin/activate

Step 3: Install dependencies

pip install -r requirements.txt

If required, install the imbalance-learning library:

pip install imbalanced-learn

Step 4: Run the EDA script

python analytics/01_eda.py

Step 5: Run the modeling script

python analytics/02_modeling.py

Step 6: Check the generated outputs

Review:

• analytics/eda_outputs/
• analytics/model_outputs/
• Saved CSV comparison tables
• Saved model pipelines
• Final model report

────────

25. Conclusion

This project demonstrates a complete data analytics and machine learning
workflow using the Titanic dataset. The process begins with dataset
profiling and exploratory analysis, followed by preprocessing,
classification, class-imbalance comparison, hyperparameter tuning,
regression, and pipeline deployment testing.

The project emphasizes reproducible preprocessing, model evaluation
using multiple metrics, and saving a complete fitted pipeline for future
predictions.
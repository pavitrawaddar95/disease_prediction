
# =====================================================
# DISEASE PREDICTION SYSTEM USING MACHINE LEARNING
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv(
    r'C:\Users\pavit\OneDrive\Desktop\disease\Disease_symptom_and_patient_profile_dataset.csv'
)

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nNUMBER OF UNIQUE DISEASES")
print(df["Disease"].nunique())

# =====================================================
# 2. DATA CLEANING
# =====================================================

# Remove duplicate records if any

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates().copy()

# Remove rows without a disease label

df = df.dropna(subset=["Disease"]).copy()

# =====================================================
# 3. EXPLORATORY DATA ANALYSIS
# =====================================================

# Disease distribution

plt.figure(figsize=(12, 6))

df["Disease"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Top 15 Common Diseases")
plt.xlabel("Disease")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Age distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=15,
    kde=True
)

plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

# Gender distribution

plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="Gender"
)

plt.title("Patient Gender Distribution")
plt.show()

# Outcome distribution

plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="Outcome Variable"
)

plt.title("Patient Outcome Distribution")
plt.show()

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

# Disease is the target variable.
# Outcome Variable is excluded because it is an outcome
# field, not a patient symptom or input measurement.

# Group diseases with fewer than 5 records.

disease_counts = df["Disease"].value_counts()

rare_diseases = disease_counts[
    disease_counts < 5
].index

df["Disease"] = df["Disease"].replace(
    rare_diseases,
    "Other / Rare Disease"
)

print("\nDISEASE DISTRIBUTION AFTER GROUPING")
print(df["Disease"].value_counts())

print("\nTotal Disease Categories:")
print(df["Disease"].nunique())

# =====================================================
# 5. SELECT INPUT FEATURES AND TARGET
# =====================================================

X = df.drop(
    ["Disease", "Outcome Variable"],
    axis=1
)

y = df["Disease"]

print("\nINPUT FEATURES")
print(X.columns.tolist())

print("\nTARGET VARIABLE")
print("Disease")

# =====================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# =====================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()

print("\nNUMERICAL FEATURES")
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)

# =====================================================
# 7. DATA PREPROCESSING
# =====================================================

numeric_transformer = Pipeline(
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

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    sparse_threshold=0
)

# =====================================================
# 8. TRAIN TEST SPLIT (80:20)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# =====================================================
# 9. MACHINE LEARNING ALGORITHMS
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

    "Naive Bayes":
        GaussianNB(),

    "Support Vector Machine":
        SVC(
            kernel="rbf",
            class_weight="balanced"
        )
}

# =====================================================
# 10. MODEL TRAINING AND EVALUATION
# =====================================================

results = {}
trained_models = {}

for name, algorithm in models.items():

    print("\n===================================")
    print("MODEL:", name)
    print("===================================")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", algorithm)
        ]
    )

    # Train model

    model.fit(X_train, y_train)

    # Make predictions

    y_pred = model.predict(X_test)

    # Evaluation metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    trained_models[name] = model

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

# =====================================================
# 11. ALGORITHM COMPARISON
# =====================================================

results_df = pd.DataFrame(results).T

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(4))

# Select the model with highest macro F1 score

best_model_name = results_df[
    "F1 Score"
].idxmax()

best_model = trained_models[
    best_model_name
]

print("\nBEST MODEL:", best_model_name)

# =====================================================
# 12. MODEL COMPARISON GRAPH
# =====================================================

results_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Algorithm Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# =====================================================
# 13. CONFUSION MATRIX
# =====================================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=best_model.classes_
)

plt.figure(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

disp.plot(
    xticks_rotation=90,
    cmap="Blues",
    ax=plt.gca(),
    colorbar=False
)

plt.title("Disease Prediction Confusion Matrix")
plt.tight_layout()
plt.show()

# =====================================================
# 14. CLASSIFICATION REPORT
# =====================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)

# =====================================================
# 15. SAVE BEST MODEL
# =====================================================

joblib.dump(
    best_model,
    "disease_prediction_model.pkl"
)

print("\nBEST MODEL SAVED SUCCESSFULLY!")

# =====================================================
# 16. PREDICT DISEASE FOR A TEST PATIENT
# =====================================================

# Demonstration using a held-out patient record

new_patient = X_test.iloc[[0]]

predicted_disease = best_model.predict(
    new_patient
)[0]

print("\nPATIENT DETAILS")
print(new_patient.to_string(index=False))

print("\nPREDICTED DISEASE:")
print(predicted_disease)

# =====================================================
# 17. PREDICT FOR A NEW PATIENT
# =====================================================

# Example patient input.
# Change these values to test different patient profiles.

new_patient_data = pd.DataFrame([{

    "Fever": "Yes",
    "Cough": "Yes",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 30,
    "Gender": "Male",
    "Blood Pressure": "Normal",
    "Cholesterol Level": "Normal"

}])

new_prediction = best_model.predict(
    new_patient_data
)

print("\nNEW PATIENT DISEASE PREDICTION")
print("--------------------------------")

print("Predicted Disease:", new_prediction[0])

print("\nPROJECT EXECUTION COMPLETED!")

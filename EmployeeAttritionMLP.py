"""
EmployeeAttritionMLP.py
------------------------
Marvellous Infosystems : Python - Automation & Machine Learning
Deep Learning Assignment - adapted to the real Employee_Attrition.csv dataset.

Written as a straight top-to-bottom SCRIPT (no functions/classes) so every task
runs in plain sequence, matching the assignment's numbered task list 1-17 plus
the 3 hyperparameter experiments.

Target: Attrition (Yes/No) -> encoded as 1/0 (Yes = employee left = positive class)

Run:
    python EmployeeAttritionMLP.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

RandomState = 42
DataPath = "data/Employee_Attrition.csv"
OutputDir = "outputs"
os.makedirs(OutputDir, exist_ok=True)

# ---------------------------------------------------------------------------
# TASK 1: Load and understand the dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 1: LOAD & UNDERSTAND THE DATASET")
print("=" * 70)

Dataset = pd.read_csv("Employee_Attrition.csv")
print(f"Shape: {Dataset.shape}")
print("\nColumn dtypes:")
print(Dataset.dtypes)
print("\nFirst 5 rows:")
print(Dataset.head())

# ---------------------------------------------------------------------------
# TASK 2: Perform exploratory analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 2: EXPLORATORY ANALYSIS")
print("=" * 70)

print("\nNumeric feature summary:")
print(Dataset.describe())

print("\nCategorical feature counts:")
for Col in ["OverTime", "Attrition"]:
    print(f"\n{Col}:")
    print(Dataset[Col].value_counts())

NumericCols = Dataset.select_dtypes(include=[np.number]).columns
CorrMatrix = Dataset[NumericCols].corr()
Fig, Ax = plt.subplots(figsize=(8, 6))
Im = Ax.imshow(CorrMatrix, cmap="coolwarm", vmin=-1, vmax=1)
Ax.set_xticks(range(len(NumericCols)))
Ax.set_yticks(range(len(NumericCols)))
Ax.set_xticklabels(NumericCols, rotation=45, ha="right")
Ax.set_yticklabels(NumericCols)
Fig.colorbar(Im, ax=Ax, label="Correlation")
Ax.set_title("Feature Correlation Heatmap")
Fig.tight_layout()
Fig.savefig(f"{OutputDir}/CorrelationHeatmap.png", dpi=150)
plt.close(Fig)
print(f"\nSaved {OutputDir}/CorrelationHeatmap.png")

# ---------------------------------------------------------------------------
# TASK 3: Find missing values
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 3: MISSING VALUES")
print("=" * 70)

MissingCounts = Dataset.isnull().sum()
MissingPct = (MissingCounts / len(Dataset) * 100).round(2)
MissingReport = pd.DataFrame({"MissingCount": MissingCounts, "MissingPct": MissingPct})
print(MissingReport)
if MissingCounts.sum() == 0:
    print("No missing values found in this dataset.")
else:
    for Col in Dataset.columns[Dataset.isnull().any()]:
        if pd.api.types.is_numeric_dtype(Dataset[Col]):
            MedianVal = Dataset[Col].median()
            Dataset[Col] = Dataset[Col].fillna(MedianVal)
            print(f"Filled missing '{Col}' with median = {MedianVal:.2f}")
        else:
            ModeVal = Dataset[Col].mode()[0]
            Dataset[Col] = Dataset[Col].fillna(ModeVal)
            print(f"Filled missing '{Col}' with mode = {ModeVal}")

# ---------------------------------------------------------------------------
# TASK 4: Check whether the target classes are balanced
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4: TARGET CLASS BALANCE")
print("=" * 70)

ClassCounts = Dataset["Attrition"].value_counts()
ClassPct = Dataset["Attrition"].value_counts(normalize=True) * 100
print(pd.DataFrame({"Count": ClassCounts, "Percent": ClassPct.round(2)}))

Fig, Ax = plt.subplots(figsize=(5, 4))
ClassCounts.plot(kind="bar", ax=Ax, color=["#4C72B0", "#DD8452"])
Ax.set_xlabel("Attrition")
Ax.set_ylabel("Count")
Ax.set_title("Target Class Balance")
Ax.tick_params(axis="x", rotation=0)
Fig.tight_layout()
Fig.savefig(f"{OutputDir}/ClassBalance.png", dpi=150)
plt.close(Fig)

ImbalanceRatio = ClassPct.max() / ClassPct.min()
if ImbalanceRatio < 1.5:
    print(f"Classes are roughly BALANCED (ratio {ImbalanceRatio:.2f}:1).")
else:
    print(f"Classes are IMBALANCED (ratio {ImbalanceRatio:.2f}:1) -> stratified split recommended.")

# ---------------------------------------------------------------------------
# TASK 5: Encode categorical variables
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 5: ENCODE CATEGORICAL VARIABLES")
print("=" * 70)

Dataset["OverTime"] = Dataset["OverTime"].map({"Yes": 1, "No": 0})
Dataset["Attrition"] = Dataset["Attrition"].map({"Yes": 1, "No": 0})

print("OverTime unique values after encoding:", Dataset["OverTime"].unique())
print("Attrition unique values after encoding:", Dataset["Attrition"].unique())

# ---------------------------------------------------------------------------
# TASK 6: Separate X and y
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 6: SEPARATE X AND y")
print("=" * 70)

X = Dataset.drop(columns=["Attrition"])
y = Dataset["Attrition"]
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# ---------------------------------------------------------------------------
# TASK 7 & 8: Split the dataset into training and testing data (stratified)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 7 & 8: TRAIN/TEST SPLIT")
print("=" * 70)

XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=RandomState, stratify=y
)
print(f"Train size: {XTrain.shape[0]}, Test size: {XTest.shape[0]}")
print(f"Train class balance:\n{yTrain.value_counts(normalize=True).round(3)}")
print(f"Test class balance:\n{yTest.value_counts(normalize=True).round(3)}")
print(
    "\nTASK 8 explanation: Yes, stratified splitting SHOULD be used here "
    f"(stratify=y is applied above). Attrition is imbalanced in this dataset "
    f"(~{ClassPct['No']:.0f}% No vs ~{ClassPct['Yes']:.0f}% Yes), so a plain random "
    "split risks pulling a test set with an even more skewed ratio, which would "
    "make accuracy look better than it really is and make precision/recall for the "
    "minority 'Yes' (attrition) class unreliable. Stratification keeps the same "
    "~80/20 class ratio in both the train and test sets, which is exactly what's "
    "needed to evaluate the model fairly on the minority class the bank/HR team "
    "actually cares about predicting."
)

# ---------------------------------------------------------------------------
# TASK 9: Scale the features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 9: FEATURE SCALING")
print("=" * 70)

Scaler = StandardScaler()
XTrainScaled = Scaler.fit_transform(XTrain)
XTestScaled = Scaler.transform(XTest)
print("Applied StandardScaler (fit on train only, applied to train and test).")

# ---------------------------------------------------------------------------
# TASK 10: Create an MLPClassifier
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 10: CREATE MLPClassifier")
print("=" * 70)

Model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=RandomState,
)


# ---------------------------------------------------------------------------
# TASK 11: Train the model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 11: TRAIN THE MODEL")
print("=" * 70)

Model.fit(XTrainScaled, yTrain)
print(f"Training complete. Iterations run: {Model.n_iter_}")

# ---------------------------------------------------------------------------
# TASKS 12-15: Accuracy, confusion matrix, classification report, P/R/F1
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASKS 12-15: EVALUATION")
print("=" * 70)

yPred = Model.predict(XTestScaled)

Accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {Accuracy:.4f}")

Cm = confusion_matrix(yTest, yPred)
print("\nConfusion Matrix:")
print(Cm)
Fig, Ax = plt.subplots(figsize=(5, 5))
Disp = ConfusionMatrixDisplay(confusion_matrix=Cm, display_labels=["Stayed", "Left"])
Disp.plot(ax=Ax, cmap="Blues", colorbar=False)
Ax.set_title("Confusion Matrix")
Fig.tight_layout()
Fig.savefig(f"{OutputDir}/ConfusionMatrix.png", dpi=150)
plt.close(Fig)

print("\nClassification Report:")
Report = classification_report(yTest, yPred, target_names=["Stayed", "Left"])
print(Report)
with open(f"{OutputDir}/ClassificationReport.txt", "w") as F:
    F.write(Report)

Precision = precision_score(yTest, yPred)
Recall = recall_score(yTest, yPred)
F1 = f1_score(yTest, yPred)
print(f"Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1-score: {F1:.4f}")

# ---------------------------------------------------------------------------
# TASK 16: Plot training loss
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 16: TRAINING LOSS CURVE")
print("=" * 70)

Fig, Ax = plt.subplots(figsize=(7, 5))
Ax.plot(Model.loss_curve_, color="#4C72B0")
Ax.set_xlabel("Iteration")
Ax.set_ylabel("Loss")
Ax.set_title("MLPClassifier Training Loss Curve")
Ax.grid(alpha=0.3)
Fig.tight_layout()
Fig.savefig(f"{OutputDir}/TrainingLoss.png", dpi=150)
plt.close(Fig)
print(f"Saved {OutputDir}/TrainingLoss.png")

# ---------------------------------------------------------------------------
# TASK 17: Test the model on new applicants (new employees)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 17: PREDICT ON NEW EMPLOYEES")
print("=" * 70)

NewEmployees = pd.DataFrame(
    [
        {  # Likely to stay: satisfied, good balance, no overtime, long tenure
            "Age": 45, "MonthlyIncome": 120000, "YearsAtCompany": 12, "TotalWorkingYears": 20,
            "DistanceFromHome": 5, "JobSatisfaction": 4, "WorkLifeBalance": 4,
            "OverTime": 0, "NumCompaniesWorked": 1, "TrainingTimesLastYear": 3,
        },
        {  # Flight risk: unhappy, overworked, far commute, job-hopper
            "Age": 26, "MonthlyIncome": 32000, "YearsAtCompany": 1, "TotalWorkingYears": 2,
            "DistanceFromHome": 28, "JobSatisfaction": 1, "WorkLifeBalance": 1,
            "OverTime": 1, "NumCompaniesWorked": 4, "TrainingTimesLastYear": 0,
        },
        {  # Middle-of-the-road employee
            "Age": 34, "MonthlyIncome": 68000, "YearsAtCompany": 4, "TotalWorkingYears": 9,
            "DistanceFromHome": 12, "JobSatisfaction": 3, "WorkLifeBalance": 3,
            "OverTime": 0, "NumCompaniesWorked": 2, "TrainingTimesLastYear": 2,
        },
    ]
)
NewEmployees = NewEmployees.reindex(columns=X.columns, fill_value=0)
NewEmployeesScaled = Scaler.transform(NewEmployees)
NewPredictions = Model.predict(NewEmployeesScaled)
NewProbabilities = Model.predict_proba(NewEmployeesScaled)[:, 1]

for i, (Pred, Prob) in enumerate(zip(NewPredictions, NewProbabilities)):
    Label = "LIKELY TO LEAVE (1)" if Pred == 1 else "LIKELY TO STAY (0)"
    print(f"Employee {i + 1}: {Label}  (P(attrition) = {Prob:.3f})")

# ---------------------------------------------------------------------------
# HYPERPARAMETER EXPERIMENTS (change one parameter at a time)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HYPERPARAMETER EXPERIMENTS")
print("=" * 70)

ExperimentResults = []

print("\n-- Experiment 1: Activation function --")
for ActivationOption in ["identity", "logistic", "tanh", "relu"]:
    ExpModel = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation=ActivationOption,
        solver="adam", max_iter=1000, random_state=RandomState,
    )
    ExpModel.fit(XTrainScaled, yTrain)
    ExpAccuracy = accuracy_score(yTest, ExpModel.predict(XTestScaled))
    print(f"  activation={ActivationOption:9s} -> accuracy={ExpAccuracy:.4f}")
    ExperimentResults.append({"Experiment": "Activation", "Value": ActivationOption, "Accuracy": ExpAccuracy})

print("\n-- Experiment 2: Hidden layer sizes --")
for LayerOption in [(10,), (20, 10), (50, 25), (100, 50, 25)]:
    ExpModel = MLPClassifier(
        hidden_layer_sizes=LayerOption, activation="relu",
        solver="adam", max_iter=1000, random_state=RandomState,
    )
    ExpModel.fit(XTrainScaled, yTrain)
    ExpAccuracy = accuracy_score(yTest, ExpModel.predict(XTestScaled))
    print(f"  hidden_layer_sizes={str(LayerOption):18s} -> accuracy={ExpAccuracy:.4f}")
    ExperimentResults.append({"Experiment": "HiddenLayers", "Value": str(LayerOption), "Accuracy": ExpAccuracy})

print("\n-- Experiment 3: learning_rate_init --")
for LearningRateOption in [0.0001, 0.001, 0.01, 0.1]:
    ExpModel = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        learning_rate_init=LearningRateOption, max_iter=1000, random_state=RandomState,
    )
    ExpModel.fit(XTrainScaled, yTrain)
    ExpAccuracy = accuracy_score(yTest, ExpModel.predict(XTestScaled))
    print(f"  learning_rate_init={LearningRateOption:<8} -> accuracy={ExpAccuracy:.4f}")
    ExperimentResults.append({"Experiment": "LearningRate", "Value": LearningRateOption, "Accuracy": ExpAccuracy})

ExperimentResultsDf = pd.DataFrame(ExperimentResults)
ExperimentResultsDf.to_csv(f"{OutputDir}/HyperparameterResults.csv", index=False)
print(f"\nSaved {OutputDir}/HyperparameterResults.csv")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Final model -> Accuracy: {Accuracy:.4f} | Precision: {Precision:.4f} | "
      f"Recall: {Recall:.4f} | F1: {F1:.4f}")
print(f"All plots/reports saved in '{OutputDir}/'")

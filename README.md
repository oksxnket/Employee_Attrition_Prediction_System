# Loan Default Prediction using a Multi-Layer Perceptron (ANN)

**Marvellous Infosystems — Python: Automation & Machine Learning — Deep Learning Assignment**

A bank wants to predict whether a loan applicant is likely to **default**, using an
Artificial Neural Network (`MLPClassifier` from scikit-learn), before approving the loan.

- `0` → Low default risk
- `1` → High default risk

## Dataset

No dataset file came with the assignment, only the feature schema below, so
`GenerateDataset.py` synthesizes a realistic 3,000-row dataset from that schema
(with a hand-crafted risk formula + noise, so the target is genuinely learnable,
plus a small percentage of injected missing values for Task 3).
If you have the bank's real dataset, just drop it in as `data/LoanDefaultDataset.csv`
with the same column names and skip running `GenerateDataset.py`.

| Feature | Description |
|---|---|
| Age | Applicant age |
| Income | Annual income |
| LoanAmount | Requested loan |
| CreditScore | Credit score |
| EmploymentYears | Years employed |
| ExistingLoans | Number of existing loans |
| MonthlyDebt | Existing monthly debt |
| LoanTerm | Loan duration |
| PreviousDefault | Yes/No |
| HomeOwnership | Rent/Own/Mortgage |
| Default | 0/1 — target |

## Project structure

```
.
├── GenerateDataset.py        # creates data/LoanDefaultDataset.csv (synthetic)
├── Main.py                   # full pipeline: EDA -> preprocessing -> train -> evaluate -> experiments
├── requirements.txt
├── data/
│   └── LoanDefaultDataset.csv
└── outputs/                  # generated on run
    ├── CorrelationHeatmap.png
    ├── ClassBalance.png
    ├── ConfusionMatrix.png
    ├── TrainingLoss.png
    ├── ClassificationReport.txt
    └── HyperparameterResults.csv
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python GenerateDataset.py       # step 1: build the dataset (skip if you have your own)
python Main.py                  # step 2: run the full assignment pipeline
```

`Main.py` prints results for every task to the console and writes plots/reports to `outputs/`.

## What `Main.py` does (mapped to the assignment tasks)

1. **Load & understand the dataset** — shape, dtypes, head.
2. **Exploratory analysis** — `describe()`, category counts, correlation heatmap.
3. **Missing values** — reports counts/percentages, imputes `Income` and
   `CreditScore` with the median.
4. **Class balance** — counts, percentages, and a bar chart.
5. **Encode categorical variables** — `PreviousDefault` (Yes/No → 1/0) label-encoded;
   `HomeOwnership` one-hot encoded (`drop_first=True`).
6. **Separate X and y**.
7. **Train/test split** — 80/20.
8. **Stratified split explanation** — the split uses `stratify=y` so the train and
   test sets keep the same default/non-default ratio as the full dataset. This
   matters most under class imbalance (a plain random split can under-represent
   the minority class in the test set by chance and distort evaluation metrics);
   it's essentially free insurance even when classes are close to balanced.
9. **Feature scaling** — `StandardScaler`, fit on train only, applied to both.
10. **Build the MLPClassifier**, starting configuration from the assignment:
    ```python
    MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
    )
    ```
11. **Train** the model on the scaled training data.
12. **Accuracy** on the held-out test set.
13. **Confusion matrix** — printed and saved as a plot.
14. **Classification report** — saved to `outputs/ClassificationReport.txt`.
15. **Precision, recall, F1-score** for the positive (default) class.
16. **Training loss curve** — plotted from `model.loss_curve_`.
17. **Predictions on new applicants** — three hand-built profiles (strong, risky,
    middle-of-the-road) run through the trained pipeline.

## Hyperparameter experiments

`RunHyperparameterExperiments()` in `Main.py` changes one parameter at a time and
reports test accuracy for each setting, saving everything to
`outputs/HyperparameterResults.csv`:

- **Experiment 1 — Activation**: `identity`, `logistic`, `tanh`, `relu`
- **Experiment 2 — Hidden layers**: `(10,)`, `(20,10)`, `(50,25)`, `(100,50,25)`
- **Experiment 3 — Learning rate**: `learning_rate_init` in `{0.0001, 0.001, 0.01, 0.1}`

Since the dataset is synthetic, exact numbers will vary run to run/dataset to dataset —
the point of the experiment section is the *methodology* (isolate one hyperparameter,
compare test accuracy, discuss over/underfitting) rather than a specific "best" number.

## Notes

- `random_state=42` is used throughout for reproducibility.
- Swap in the bank's real data by replacing `data/LoanDefaultDataset.csv` — the
  pipeline doesn't need any other changes as long as the column names match.

---
*Piyush Khairnar — Marvellous Infosystems*

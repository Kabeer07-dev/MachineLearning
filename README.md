# 🤖 Machine Learning Projects

A collection of machine learning projects built with Python and Jupyter Notebooks. Each project takes a real-world dataset through EDA, cleaning, feature engineering, and model training/evaluation.

---

## 📁 Repository Structure

```
MachineLearning/
│
├── Heart_Disease_Prediction/
│   ├── heart.csv           # Dataset
│   └── heart.ipynb         # EDA + Random Forest classifier
│
├── Insurance_prediction/
│   ├── insurance.csv       # Dataset
│   └── EDA_insurance.ipynb # EDA + Linear Regression model
│
├── Titanic_Survival_Prediction/
│   ├── titanic.csv         # Dataset
│   └── titanic.ipynb       # EDA + Logistic Regression classifier
│
└── README.md
```

---

## 📓 Projects

### 1. 🫀 Heart Disease Prediction

**Folder:** [`Heart_Disease_Prediction/`](./Heart_Disease_Prediction) · **Notebook:** `heart.ipynb` · **Dataset:** `heart.csv`

Predicts the likelihood of heart disease from clinical features.

**Key steps:**
- EDA with histograms, count plots by sex/chest pain type/fasting blood sugar, and a correlation heatmap
- Cleaning invalid zero-values in `Cholesterol` and `RestingBP` (replaced with column mean)
- One-hot encoding of categorical features and `StandardScaler` feature scaling
- `RandomForestClassifier` tuned with `GridSearchCV` over `n_estimators`, `max_depth`, and `criterion`

**Result:** ~87% test accuracy with the tuned Random Forest model (best params: `criterion='entropy'`, `max_depth=None`, `n_estimators=200`).

---

### 2. 📊 Insurance Charges Prediction

**Folder:** [`Insurance_prediction/`](./Insurance_prediction) · **Notebook:** `EDA_insurance.ipynb` · **Dataset:** `insurance.csv`

Explores how age, BMI, smoking status, and region relate to insurance charges, then fits a regression model.

**Key steps:**
- EDA: distribution plots, box plots for outlier detection, correlation heatmap
- Label encoding (`sex`, `smoker`) and one-hot encoding (`region`)
- Feature engineering: BMI bucketed into `Underweight` / `Normal` / `Overweight` / `Obese` categories
- Pearson correlation analysis against `charges`
- `LinearRegression` model to predict insurance charges

**Result:** R² score of ~0.80 on the test set.

---

### 3. 🚢 Titanic Survival Prediction

**Folder:** [`Titanic_Survival_Prediction/`](./Titanic_Survival_Prediction) · **Notebook:** `titanic.ipynb` · **Dataset:** `titanic.csv`

Predicts passenger survival on the Titanic from demographic and ticket features.

**Key steps:**
- EDA: survival rate by gender and passenger class, distribution plots for `Age` and `Fare`
- Missing-value handling (`Age` → median, `Embarked` → mode, `Cabin` dropped) and IQR-based outlier check on `Fare`
- Encoding (`Sex` → `is_female`, one-hot `Embarked`) and `StandardScaler` scaling
- Feature engineering: title extracted from `Name` (grouped into `Mr`/`Mrs`/`Miss`/`Rare`), `FamilySize`, and `IsAlone`
- `LogisticRegression` classifier

**Result:** ~80.45% test accuracy.

---

## 🛠️ Tech Stack

| Tool                 | Purpose                              |
| --------------------- | ------------------------------------ |
| Python 3.x            | Core programming language            |
| Jupyter Notebook      | Interactive development environment  |
| Pandas / NumPy        | Data manipulation and numerical ops  |
| Matplotlib / Seaborn  | Data visualization                   |
| Scikit-learn          | Preprocessing, models, evaluation    |
| SciPy                 | Pearson correlation analysis         |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Run the notebooks

```bash
# Clone the repository
git clone https://github.com/Kabeer07-dev/MachineLearning.git

# Navigate to the project directory
cd MachineLearning

# Launch Jupyter Notebook
jupyter notebook
```

Each project folder is self-contained — open the `.ipynb` file inside it and run the cells top to bottom. The datasets are already included in each folder, so no separate download step is needed.

---

## 📌 Topics Covered

- Exploratory Data Analysis (EDA)
- Data Cleaning & Missing-Value Handling
- Outlier Detection (IQR method)
- Feature Engineering & Encoding (label, one-hot)
- Feature Scaling
- Classification (Random Forest, Logistic Regression)
- Regression (Linear Regression)
- Hyperparameter Tuning (GridSearchCV)
- Model Evaluation (accuracy, R²)

---

## 👤 Author

**Kabeer** — [@Kabeer07-dev](https://github.com/Kabeer07-dev)

---

## 📄 License

No license file is currently included in this repository, so all rights are reserved by default. If you'd like others to freely use or build on this code, consider adding a [license](https://choosealicense.com/) such as MIT.

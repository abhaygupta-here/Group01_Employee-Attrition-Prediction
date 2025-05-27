# Group01_Employee-Attrition-Prediction


This project aims to predict whether an employee is likely to leave a company using historical HR analytics data from IBM. We apply classification techniques and explore feature importance to gain actionable insights.

## 📁 Dataset

- **Source**: [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-datase)
- **Format**: CSV
- **Target Variable**: `Attrition` (Yes/No)

## 🧠 Problem Statement

To build a machine learning model that predicts employee attrition and helps organizations identify key factors influencing retention.

---

## 🔧 Methodology

### Data Preprocessing
- Dropped non-informative columns: `EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount`
- Encoded categorical variables using `LabelEncoder`
- Split into features (`X`) and target (`y`)
- 80-20 train-test split

### Model Building
- Model used: `RandomForestClassifier`
- Evaluation using:  
  - Classification Report  
  - Confusion Matrix  
  - Feature Importance Plot

---

## 💻 Code Example

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

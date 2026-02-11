# 🩺 Diabetes Prediction using Logistic Regression (From Scratch)

## 📌 Project Overview

This project implements **Logistic Regression from scratch** using Gradient Descent on the Diabetes dataset.

The notebook includes:

- Data preprocessing
- Handling missing values
- Train-test split
- Feature scaling
- Sigmoid function implementation
- Binary Cross-Entropy loss
- Gradient Descent optimization
- Accuracy evaluation

No sklearn LogisticRegression model is used.

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

---

# 📂 Dataset Loading

```python
df = pd.read_csv("diabetes.csv")
df.head()
```

Target column:
- `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

# 🧹 Data Preprocessing

## Handle Missing Values

Certain columns contain 0 values which are invalid (e.g., BMI, BloodPressure).

```python
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())
```

---

# 🎯 Feature Matrix & Target Vector

```python
X = df.drop("Outcome", axis=1).values
Y = df["Outcome"].values.reshape(-1,1)
```

---

# 📊 Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```

---

# 📏 Feature Scaling (Standardization)

```python
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
```

Formula:

```
Z = (X − μ) / σ
```

---

# 🔢 Logistic Regression from Scratch

---

## 1️⃣ Sigmoid Function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Mathematical Formula:

```
σ(z) = 1 / (1 + e^(-z))
```

---

## 2️⃣ Binary Cross-Entropy Loss

```python
def compute_loss(y, y_hat):
    m = len(y)
    return -(1/m) * np.sum(
        y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    )
```

Formula:

```
Loss = - (1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]
```

---

## 3️⃣ Gradient Descent

```python
def train(X, Y, lr=0.01, epochs=1000):

    m, n = X.shape
    w = np.zeros((n,1))
    b = 0

    for i in range(epochs):

        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_hat - Y))
        db = (1/m) * np.sum(y_hat - Y)

        w -= lr * dw
        b -= lr * db

    return w, b
```

---

# 🔮 Prediction Function

```python
def predict(X, w, b):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    return (y_hat > 0.5).astype(int)
```

Threshold:
- If probability > 0.5 → Class 1
- Else → Class 0

---

# 📊 Model Training & Evaluation

```python
w, b = train(X_train, Y_train)

y_pred = predict(X_test, w, b)

accuracy = np.mean(y_pred == Y_test)

print("Accuracy:", accuracy)
```

Accuracy Formula:

```
Accuracy = Correct Predictions / Total Predictions
```

---

# 🧠 Concepts Used

- Logistic Regression
- Sigmoid Activation
- Binary Classification
- Cross-Entropy Loss
- Gradient Descent
- Feature Scaling
- Train-Test Split


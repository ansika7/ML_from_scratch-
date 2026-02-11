# ❤️ Heart Disease Prediction using Linear Regression (From Scratch)

## 📌 Project Overview

This project implements **Linear Regression from scratch using Mini-Batch Gradient Descent** on the UCI Heart Disease dataset.

The notebook includes:

- Data preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split
- Manual gradient descent implementation
- Mean Squared Error (MSE) calculation

No sklearn model is used for training — only NumPy.

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
df1 = pd.read_csv("heart_disease_uci.csv")
df = df1
```

---

# 📊 Data Exploration

```python
df.columns
df.describe()
df.info()
```

Value counts for categorical columns:

```python
df["thalch"].value_counts()
df["restecg"].value_counts()
df["fbs"].value_counts()
df["exang"].value_counts()
df["cp"].value_counts()
```

---

# 🧹 Data Cleaning

### Drop Unnecessary Columns

```python
df = df.drop(columns=["ca","thal","slope","dataset"])
df = df.drop(columns="id")
```

---

### Handle Missing Values

```python
df["trestbps"] = df["trestbps"].fillna(np.mean(df["trestbps"]))
df["chol"] = df["chol"].fillna(np.mean(df["chol"]))
df["thalch"] = df["thalch"].fillna(np.mean(df["thalch"]))
df["oldpeak"] = df["oldpeak"].fillna(np.mean(df["oldpeak"]))

df["restecg"] = df["restecg"].fillna("normal")
df["fbs"].fillna(0, inplace=True)
df["exang"] = df["exang"].fillna(0)
```

---

# 🔢 Encoding Categorical Variables

```python
encode = {
    "restecg":{"normal":0,"lv hypertrophy":1,"st-t abnormality":2},
    "fbs":{False:0, True:1},
    "exang":{False:0, True:1},
    "sex":{"Male":1,"Female":0},
    "num":{0:0,(1,2,3,4):1},
    "cp":{'asymptomatic':0,'typical angina':1,'non-anginal':2,'atypical angina':3}
}

df = df.replace(encode)
```

Target variable:
- `num` → 0 (No Heart Disease)
- 1 (Heart Disease)

---

# 📏 Feature Scaling (Standardization)

Manual Z-score normalization:

```python
df["trestbps"] = df["trestbps"].apply(lambda s: (s - df["trestbps"].mean()) / df["trestbps"].std())
df["age"] = df["age"].apply(lambda s: (s - df["age"].mean()) / df["age"].std())
df["chol"] = df["chol"].apply(lambda s: (s - df["chol"].mean()) / df["chol"].std())
df["thalch"] = df["thalch"].apply(lambda s: (s - df["thalch"].mean()) / df["thalch"].std())
df["oldpeak"] = df["oldpeak"].apply(lambda s: (s - df["oldpeak"].mean()) / df["oldpeak"].std())
```

Formula:

```
Z = (X − μ) / σ
```

---

# 🎯 Train-Test Split

```python
target = df["num"]
features = df.iloc[:,:-1]

X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, train_size=0.8,
    shuffle=True, random_state=42
)

X_train = np.array(X_train, dtype=float)
Y_train = np.array(Y_train, dtype=float).reshape(-1,1)

X_test = np.array(X_test, dtype=float)
Y_test = np.array(Y_test, dtype=float)
```

---

# 📉 Gradient Descent Implementation

## Gradient Calculation

```python
def gradient_descent1(x, y, y_hat):
    m = x.shape[0]
    dw = (1/m) * np.dot(x.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db
```

---

## Training Function (Mini-Batch Gradient Descent)

```python
def train_linear():
    bs = 40
    lr = 0.001
    epoch = 150
    m, n = X_train.shape

    w = np.zeros((n,1))
    b = 0

    for i in range(epoch):
        for j in range((m-1)//bs + 1):

            start_indx = j * bs
            end_indx = start_indx + bs

            x = X_train[start_indx:end_indx]
            y = Y_train[start_indx:end_indx]

            y_hat = np.dot(x, w) + b
            dw, db = gradient_descent1(x, y, y_hat)

            w -= lr * dw
            b -= lr * db

    return w, b
```

---

# 🔮 Prediction

```python
def predict(w, b):
    y_hat = np.dot(X_test, w) + b
    return y_hat
```

---

# 📊 Model Evaluation

```python
w, b = train_linear()
y_pred = predict(w, b)

loss = np.mean((Y_test - y_pred)**2)
print(loss)
```

Loss Function:

```
MSE = (1/m) Σ (y - ŷ)²
```

---

# 🧠 Concepts Used

- Linear Regression
- Mini-Batch Gradient Descent
- Z-score Normalization
- Encoding categorical features
- Mean Squared Error
- Train-Test Split


# 🌳 Decision Tree Classifier (From Scratch)

## 📌 Project Overview

This project implements a **Decision Tree Classifier from scratch** using:

- NumPy
- Pandas
- Gini Index (for split criterion)

The model is trained and tested on the **Iris dataset**.

---

## 🛠️ Libraries Used

- NumPy
- Pandas

Install dependencies:

```bash
pip install numpy pandas
```

---

# 📂 Implementation Details

---

## 1️⃣ Import Libraries

```python
import numpy as np 
import pandas as pd
```

---

## 2️⃣ Load Dataset

```python
df = pd.read_csv("/home/mllab/IRIS.csv")
df.info()
```

Dataset: Iris  
Features:
- sepal_length
- sepal_width
- petal_length
- petal_width

Target:
- species

---

## 3️⃣ Feature Matrix and Target Variable

```python
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y = df["species"].values
```

- `X` → Feature matrix  
- `Y` → Target labels  

---

## 4️⃣ Train-Test Split (80-20)

```python
idx = int(0.8 * len(X))

X_train = X[:idx]
X_test = X[idx:]
Y_train = Y[:idx]
Y_test = Y[idx:]
```

- 80% training data  
- 20% testing data  

---

# 🌿 Gini Index

## 5️⃣ Gini Function

```python
def gini(y):
    count = np.unique(y)
    g = 1.0
    for c in count:
        p = (np.sum(y == c)) / len(y)
        g = g - p**2
    return g
```

### Formula:

Gini = 1 − Σ (pᵢ²)

Where:
- pᵢ = probability of class i

Lower Gini → Better split

---

# 🔎 Finding Best Split

## 6️⃣ Best Split Function

```python
def best_split(X, Y):
    best = 1
    split_threshold = None
    split_feature = None

    for feature in range(len(X[0])):
        threshold = np.unique(X[:, feature])

        for t in threshold:
            left = Y[X[:, feature] <= t]
            right = Y[X[:, feature] > t]

            g = (len(left)/len(Y)) * gini(left) + \
                (len(right)/len(Y)) * gini(right)

            if g < best:
                best = g
                split_threshold = t
                split_feature = feature

    return split_feature, split_threshold
```

This function:
- Tries all features
- Tries all possible thresholds
- Selects the split with minimum Gini impurity

---

# 🌳 Decision Tree Structure

## 7️⃣ Node Class

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
```

Each node stores:
- Feature index
- Threshold
- Left subtree
- Right subtree
- Class value (if leaf)

---

# 🌲 Build Tree Recursively

## 8️⃣ Tree Construction

```python
def build_tree(X, Y, depth=0, max_dep=5):

    if len(np.unique(Y)) == 1 or depth == max_dep:
        return Node(value=max(set(Y), key=list(Y).count))

    feature, threshold = best_split(X, Y)

    if feature is None:
        return Node(value=max(set(Y), key=list(Y).count))

    left = X[:, feature] <= threshold
    right = X[:, feature] > threshold

    left_child = build_tree(X[left], Y[left], depth+1, max_dep=5)
    right_child = build_tree(X[right], Y[right], depth+1, max_dep=5)

    return Node(feature, threshold, left_child, right_child)
```

Stopping conditions:
- All samples belong to one class
- Maximum depth reached

---

# 🔮 Prediction

## 9️⃣ Predict One Sample

```python
def predict_one(x, tree):
    if tree.value is not None:
        return tree.value

    if x[tree.feature] <= tree.threshold:
        return predict_one(x, tree.left)
    else:
        return predict_one(x, tree.right)
```

---

## 🔟 Predict Multiple Samples

```python
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])
```

---

# 📊 Model Training & Evaluation

```python
tree = build_tree(X_train, Y_train, depth=0, max_dep=5)

y_pred = predict(X_test, tree)

acc = np.mean(y_pred == Y_test)

print("Accuracy", acc)
```

Accuracy is calculated as:

Accuracy = Correct Predictions / Total Predictions

---

# 🎯 Key Concepts Used

- Decision Tree
- Gini Index
- Recursive Tree Building
- Train-Test Split
- Classification Accuracy

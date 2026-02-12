# 🚗 PCA From Scratch on Vehicle Dataset

## 📌 Project Overview

This project implements **Principal Component Analysis (PCA) from scratch** using only:

- NumPy
- Pandas
- Matplotlib

No Scikit-learn PCA was used.

The goal of this project is to:
- Preprocess real-world vehicle data
- Handle missing values
- Encode categorical features
- Standardize the dataset
- Compute covariance matrix manually
- Compute eigenvalues and eigenvectors using Power Iteration
- Reduce dimensions to 2 principal components
- Visualize the transformed data

---

## 📂 Dataset

Dataset used: **Car Details Dataset (Version 4)**

The dataset contains:
- Engine details
- Power & torque specifications
- Dimensions
- Fuel tank capacity
- Drivetrain type
- Seating capacity
- And other vehicle specifications

---

## 🧠 Theory Behind PCA

### What is PCA?

Principal Component Analysis (PCA) is a **dimensionality reduction technique** that transforms data into a new coordinate system such that:

- The first principal component captures the maximum variance.
- Each succeeding component captures the remaining variance.
- All components are orthogonal (uncorrelated).

---

### Mathematical Steps

1. Standardize the dataset

\[
Z = \frac{X - \mu}{\sigma}
\]

2. Compute Covariance Matrix

\[
C = \frac{X^T X}{n-1}
\]

3. Solve Eigenvalue Problem

\[
Cv = \lambda v
\]

Where:
- \( \lambda \) = eigenvalue (variance explained)
- \( v \) = eigenvector (principal direction)

4. Project Data

\[
X_{PCA} = XW
\]

Where:
- \( W \) contains top k eigenvectors

---

## ⚙️ Implementation Steps

### 1️⃣ Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

### 2️⃣ Load Dataset

```python
df = pd.read_csv("car details v4.csv")
df.info()
```

---

### 3️⃣ Handle Missing Values

Categorical columns:

```python
df["Engine"] = df["Engine"].fillna("1197cc")
df["Max Power"] = df["Max Power"].fillna("89 bhp @ 4000 rpm")
df["Max Torque"] = df["Max Torque"].fillna("200 Nm @ 1750 rpm ")
df["Drivetrain"] = df["Drivetrain"].fillna("FWD")
```

Numerical columns:

```python
df["Length"] = df["Length"].fillna(np.mean(df["Length"]))
df["Width"] = df["Width"].fillna(np.mean(df["Width"]))
df["Height"] = df["Height"].fillna(np.mean(df["Height"]))
df["Seating Capacity"] = df["Seating Capacity"].fillna(np.mean(df["Seating Capacity"]))
df["Fuel Tank Capacity"] = df["Fuel Tank Capacity"].fillna(np.mean(df["Fuel Tank Capacity"]))
```

---

### 4️⃣ Encode Categorical Variables

```python
col = df.columns
dt = df.dtypes.values

for i in range(len(col)):
    if dt[i] == 'object':
        df[col[i]] = df[col[i]].astype('category').cat.codes
```

---

### 5️⃣ Standardize Data

```python
for i in range(len(col)):
    if dt[i] != 'object':
        X = df[col[i]]
        df[col[i]] = (X - np.mean(X)) / np.std(X)
```

---

### 6️⃣ Compute Covariance Matrix

```python
X = df
n = X.shape[0]
cov_matrix = np.dot(X.T, X) / (n - 1)
```

---

### 7️⃣ Eigen Decomposition (From Scratch)

#### Power Iteration

```python
def power_iteration(A, iterations=1000, tol=1e-6):
    n = A.shape[0]
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)

    for _ in range(iterations):
        b_new = A @ b
        b_new = b_new / np.linalg.norm(b_new)
        if np.linalg.norm(b - b_new) < tol:
            break
        b = b_new

    eigenvalue = b.T @ A @ b
    return eigenvalue, b
```

#### Deflation

```python
def deflate(A, eigenvalue, eigenvector):
    return A - eigenvalue * np.outer(eigenvector, eigenvector)
```

#### Full Decomposition

```python
def manual_eigen_decomposition(A, k):
    A_copy = A.copy()
    eigenvalues = []
    eigenvectors = []

    for _ in range(k):
        val, vec = power_iteration(A_copy)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        A_copy = deflate(A_copy, val, vec)

    return np.array(eigenvalues), np.array(eigenvectors).T
```

---

### 8️⃣ Reduce to 2 Principal Components

```python
k = 2
eigenvalues, eigenvectors = manual_eigen_decomposition(cov_matrix, k)

X_pca = X @ eigenvectors
X_pca = np.array(X_pca)
```

---

### 9️⃣ Visualization

```python
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA from Scratch on Vehicle Dataset")
plt.show()
```

---

## 📊 Results

- Data successfully reduced from high-dimensional space to 2D.
- Principal components capture maximum variance directions.
- Visualization shows distribution of vehicles in reduced space.

---

## 🎯 Key Learning Outcomes

✔ Implemented PCA without Scikit-learn  
✔ Understood covariance matrix computation  
✔ Implemented Power Iteration method  
✔ Applied eigenvalue deflation  
✔ Built dimensionality reduction pipeline from scratch  

---

## 🚀 Future Improvements

- Compare with Sklearn PCA
- Add explained variance ratio
- Improve categorical encoding
- Perform clustering after PCA
- Add scree plot visualization

---

## 📚 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib

---

## ⭐ Author

**ANSIKA KUNDU**

Machine Learning Enthusiast | AI Student

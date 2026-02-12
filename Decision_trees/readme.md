# Decision Tree Classifier From Scratch (IRIS Dataset)

## 📌 Project Overview

This project implements a **Decision Tree Classifier from scratch** using only **NumPy and Pandas**, without using scikit-learn or any built-in ML models.

The model is trained and tested on the **IRIS dataset** to classify flower species.

---

## 🎯 Objective

To manually build a Decision Tree classifier using:

- Gini Index as the splitting criterion  
- Recursive tree construction  
- Custom prediction logic  
- Manual accuracy evaluation  

---

## 📂 Dataset

**Dataset:** IRIS.csv  

### Features:
- sepal_length  
- sepal_width  
- petal_length  
- petal_width  

### Target:
- species  

---

## ⚙️ Implementation Steps

### 1️⃣ Data Loading
- Loaded dataset using Pandas  
- Separated features (X) and target (Y)  

---

### 2️⃣ Train-Test Split
- Manual 80–20 split  
- No sklearn utilities used  

---

### 3️⃣ Gini Impurity Calculation

Gini Formula:

Gini = 1 − Σ(p²)

Where `p` represents the probability of each class in a node.

Used to measure node impurity and determine best splits.

---

### 4️⃣ Best Split Function

- Iterates through all features  
- Tests possible threshold values  
- Calculates weighted Gini impurity  
- Selects feature & threshold with lowest impurity  

---

### 5️⃣ Tree Structure

Custom `Node` class stores:

- Splitting feature  
- Threshold value  
- Left child  
- Right child  
- Leaf node prediction  

---

### 6️⃣ Recursive Tree Building

Stopping conditions:

- All samples belong to one class  
- Maximum depth reached  
- No valid split found  

---

### 7️⃣ Prediction Logic

- Recursive traversal of the tree  
- Returns class label at leaf node  

---

## 📊 Model Evaluation

Accuracy is calculated as:

Accuracy = (Correct Predictions) / (Total Predictions)

Final accuracy is printed after testing on the test dataset.

---

## 🛠 Tech Stack

- Python  
- NumPy  
- Pandas  

---

## 🚀 Key Learning Outcomes

- Understanding Gini Impurity  
- Implementing tree-based algorithms from scratch  
- Recursive algorithm design  
- Manual model evaluation  
- Core machine learning fundamentals  
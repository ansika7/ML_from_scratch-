# Multinomial Naive Bayes From Scratch

A simple implementation of the Multinomial Naive Bayes classifier built from first principles using NumPy and Pandas.

---

## Overview

This project implements a probabilistic text classification model without using high-level machine learning libraries.  
All priors, likelihoods, and predictions are computed manually.

The model supports:

- Count-based feature vectors (Bag-of-Words)
- Laplace smoothing
- Log-probability computation for numerical stability
- Multi-class classification

---

## Mathematical Model

Posterior probability:

P(C_k | X) ∝ P(C_k) ∏ P(x_i | C_k)^{x_i}

Smoothed likelihood:

P(x_i | C_k) = (N_ik + 1) / (N_k + V)

All computations are performed in log-space during prediction.

---

## Tech Stack

- Python 3  
- NumPy  
- Pandas  

---

## Project Structure


---

## Usage

```python
model = MultinomialNaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

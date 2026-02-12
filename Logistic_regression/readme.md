# 🐶🐱 CNN-Based Image Classification (Binary Classification)

A Convolutional Neural Network (CNN) built using **TensorFlow/Keras** for binary image classification.  
The model is trained on a structured image dataset and evaluates performance using accuracy and loss curves.

---

## 📌 Project Overview

This project implements an end-to-end deep learning pipeline:

- 📂 Dataset loading using `image_dataset_from_directory`
- 🎨 Manual grayscale preprocessing using weighted RGB conversion
- 🧠 CNN model design using Conv2D + MaxPooling layers
- ⚙️ Model training with Adam optimizer
- 📊 Performance visualization (Accuracy & Loss curves)

---

## 🗂 Dataset Structure

The dataset directory should follow this structure:

```
training_set/
    class_1/
    class_2/

test_set/
    class_1/
    class_2/
```

Labels are inferred automatically from folder names.

---

## ⚙️ Dependencies

```bash
pip install tensorflow matplotlib
```

---

## 🧹 Data Preprocessing

### 1️⃣ Image Resizing
All images are resized to:

```
(180, 180)
```

### 2️⃣ Normalization
Pixel values are scaled using:

```python
layers.Rescaling(1./255)
```

### 3️⃣ Manual Grayscale Conversion

Weighted grayscale formula used:

\[
Gray = 0.299R + 0.587G + 0.114B
\]

```python
def rgb_to_grayscale_weighted(x):
    weights = K.constant([[[[0.299, 0.587, 0.114]]]])
    return K.sum(x * weights, axis=-1, keepdims=True)
```

---

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Rescaling | 1./255 normalization |
| Conv2D | 16 filters, 3x3, ReLU |
| MaxPooling | 2x2 |
| Conv2D | 32 filters, 3x3, ReLU |
| MaxPooling | 2x2 |
| Conv2D | 64 filters, 3x3, ReLU |
| MaxPooling | 2x2 |
| Flatten | - |
| Dense | 128 units, ReLU |
| Output | 2 units (logits) |

### Model Compilation

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

---

## 🚀 Training

```python
epochs = 5

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

---

## 📊 Performance Visualization

The following metrics are plotted:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

```python
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
```

---

## 📈 Expected Output

- Increasing training accuracy
- Stable validation accuracy
- Decreasing loss curves
- Insight into overfitting/underfitting behavior

---

## 🧪 Research Notes

- Uses Sparse Categorical Crossentropy with logits
- CNN depth increased progressively (16 → 32 → 64 filters)
- No data augmentation used (can be added for improvement)
- Grayscale transformation implemented manually for understanding

---



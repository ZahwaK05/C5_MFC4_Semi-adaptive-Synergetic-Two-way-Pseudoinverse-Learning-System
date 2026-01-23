# Semi-Adaptive Synergetic Two-Way Pseudoinverse Learning System

This repository implements a **deep learning framework without gradient descent**, based entirely on **pseudoinverse learning**.  
The proposed system performs **forward learning, backward learning, feature fusion, and ensemble learning** using **closed-form analytical solutions** instead of backpropagation.

The framework is designed for **fast, stable, and globally optimal training**, and is experimentally validated on the **MNIST handwritten digit dataset**.

---

## 🚀 Key Contributions

- ✅ Deep learning **without backpropagation**
- ✅ Closed-form training using **pseudoinverse**
- ✅ **Two-way learning** (forward + backward)
- ✅ **Feature fusion** of input-driven and label-driven representations
- ✅ **Ensemble learning** for improved robustness
- ✅ High accuracy with low computational complexity

---

## 🧠 Method Overview

The system consists of four major learning stages:

### 1. Forward Learning (PILAE)
- Uses **Pseudoinverse Learning Autoencoders (PILAE)**
- Extracts hierarchical features layer by layer
- Encoder and decoder weights are computed analytically

### 2. Classification (SHLNN)
- A **Single Hidden Layer Neural Network**
- Output weights are computed using a closed-form least squares solution
- No learning rate, epochs, or gradient descent

### 3. Backward Learning
- Propagates **label information backward** into hidden layers
- Uses **inverse activation functions** and regularized pseudoinverse
- Refines feature representations with class-discriminative information

### 4. Feature Fusion & Ensemble Learning
- Forward and backward features are concatenated
- A fusion classifier is trained analytically
- Multiple subnetworks are combined using ensemble averaging




## 📂 Repository Structure


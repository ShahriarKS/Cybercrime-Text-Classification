# Cybercrime Text Classification & Triaging System

An advanced, multi-model Natural Language Processing (NLP) project designed to classify cybercrime complaints into two critical categories:
- **Financial Fraud**
- **Non-Financial Cybercrime**

The project scales from classical baseline Machine Learning models to State-of-the-Art Deep Learning Transformer architectures to handle severe class imbalance using analytical loss weighting.

## 📂 Project Structure

This repository contains the following implementation notebooks:
1. [cyberCrimePythonBasic.ipynb](./notebook/cyberCrimePythonBasic.ipynb): Baseline Classical ML Pipelines (TF-IDF + Naive Bayes, Logistic Regression, SGD, Linear SVM).
2. [distilbertAndRoberta.ipynb](./notebook/distilbertAndRoberta.ipynb): Advanced Transformer Pipelines fine-tuned on GPU (DistilBERT & RoBERTa-base with Custom Weighted Loss).

## 📊 Dataset

- **Total Samples:** 85,876 (Split into Training, Validation, and Unseen Test sets)
- **Source:** Cybercrime Complaint Dataset

## 🚀 Preprocessing & Feature Engineering

1. **Classical Pipeline:** Missing value removal, duplicate elimination, specialized text cleaning, and structural **TF-IDF Vectorization**.
2. **Deep Learning Pipeline:** Context-aware sub-word tokenization utilizing WordPiece (for DistilBERT) and Byte-level BPE (for RoBERTa) with a fixed maximum sequence constraint of 512 tokens.

---

## 📈 Results & Performance Evaluation

All models were evaluated across an extensive unseen test dataset containing **17,176 complaints**. To address data imbalance, custom class weights (`[1.5, 1.0]`) were dynamically injected into the Transformer architectures.

| Model Class | Model Architecture | Overall Accuracy | Non-Financial Precision | Financial Fraud Recall | Macro Avg F1-Score | Training Framework |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Traditional ML** | Naive Bayes | 81.97% | - | - | - | CPU Baseline |
| **Traditional ML** | SGD Classifier | 81.43% | - | - | - | CPU Baseline |
| **Traditional ML** | Linear SVM | 81.83% | - | - | - | CPU Baseline |
| **Traditional ML** | Logistic Regression | 82.35% |- | - |- | CPU Baseline |
| **Deep Learning** | **DistilBERT** | **84.00%** | 0.77 | 0.85 | 0.83 | 12 Mins (T4 GPU) |
| **Deep Learning** | **RoBERTa-base** | **84.00%** | **0.79** | **0.86** | **0.83** | 35 Mins (T4 GPU) |

### Key Analytical Insights:
- **The Transformer Advantage:** While global accuracy remains structurally constrained at 84% due to text overlaps, Deep Learning models vastly outperformed Classical ML in core business metrics.
- **RoBERTa-base (Best Production Variant):** Delivers a **5% reduction in false alarms** (Precision: 0.79) for minority classes and catches **5% more actual fraud cases** (Recall: 0.86) compared to baseline Logistic Regression.
- **DistilBERT (Best Efficiency Variant):** Achieves near-identical accuracy to RoBERTa while offering a **3x faster training execution loop** (12 minutes vs 35 minutes).

---

---

# 📊 Visualizations

The following visualizations summarize dataset analysis, model performance, and evaluation results of the cybercrime text classification pipeline.

## 1. Exploratory Data Analysis & Dataset Insights

### Class Distribution

![Binary Category](./images/binary%20category.png)

### Dataset Splits

![Train Test Split](./images/test%20train%20split.png)

![Train and Test Dataset Split](./images/test%20and%20train%20dataset%20split.png)

---

## 2. Baseline Machine Learning Performance

### Overall Model Comparison

![Comparison of Machine Learning Models](./images/Comparison%20of%20Machine%20Learning%20Models.png)

### Confusion Matrix (Best Model — SVM)

![Confusion Matrix](./images/confusion%20matrix.png)

### Individual Model Performance

#### Linear SVM

![Linear SVM Performance](./images/linear%20SVM%20performance.png)

#### Logistic Regression

![Logistic Regression Performance](./images/Logistic%20Regression%20Performance.png)

#### Naive Bayes

![Naive Bayes Performance](./images/naive%20bayes%20performance.png)

#### SGD Classifier

![SGD Performance](./images/SGD%20performance.png)

---

## 3. Advanced Transformer Deep Learning Diagnostics

### DistilBERT Engine Trends

#### Training vs Validation Loss

![Training vs Validation Loss](./images/Training%20vs%20Validation%20Loss.png)

#### Validation Accuracy & F1-Score

![DistilBERT Validation Accuracy & F1-Score](./images/DistilBERT%20Validation%20Accuracy%20%26%20F1-Score.png)

---

### RoBERTa-base Diagnostics

#### Training vs Validation Loss

![RoBERTa-base Training vs Validation Loss](./images/RoBERTa-base%20Training%20vs%20Validation%20Loss%20Across%20Epochs.png)

#### Validation Accuracy & F1-Score

![RoBERTa-base Validation Accuracy & F1-Score](./images/RoBERTa-base%20Validation%20Accuracy%20%26%20F1-Score%20Trend.png)

---

### Transformer Benchmark Analysis

![Transformer Models Performance Comparison](./images/Transformer%20Models%20Performance%20Comparison%20(di...).png)

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/ShahriarKS/Cybercrime-Text-Classification.git](https://github.com/ShahriarKS/Cybercrime-Text-Classification.git)

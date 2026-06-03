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

## 📊 Visualizations

The complete exploratory data analysis (EDA), training diagnostics, and model comparison graphs are structured below. All original plots can be found under the `images/` directory.

### 1. Exploratory Data Analysis & Dataset Insights
Graphs showcasing the initial dataset distribution and the balancing splits between training, validation, and testing setups.
* **Class Distribution:** ![Binary Category](images/binary_category.png)
* **Dataset Splits:** ![Test Train Split](images/test_train_split.png) | ![Test and Train Dataset Split](images/test_and_train_dataset_split.png)

### 2. Baseline Machine Learning Performance
Visual performance and error metrics of the traditional classical models.
* **Overall ML Comparison:** ![Comparison of Machine Learning Models](images/Comparison_of_Machine_Learning_Models.png)
* **Confusion Matrix:** ![Confusion Matrix](images/confusion_matrix.png)
* **Individual Model Plots:** 
  * ![Linear SVM Performance](images/Linear_SVM_Performance.png)
  * ![Logistic Regression Performance](images/Logistic_Regression_Performance.png)
  * ![Naive Bayes Performance](images/Naive_Bayes_Performance.png)
  * ![SGD Performance](images/SGD_performance.png)

### 3. Advanced Transformers Deep Learning Diagnostics
Training logs tracking the optimization convergence, learning curves, and performance progress of the deep learning pipelines across validation cycles.

#### DistilBERT Engine Trends:
* **Training vs Validation Loss:** ![Training vs Validation Loss](images/Training_vs_Validation_Loss.png)
* **Validation Accuracy & F1-Score:** ![DistilBERT Validation Accuracy & F1-Score](images/DistilBERT_Validation_Accuracy_&_F1-Score.png)

#### RoBERTa-base Engine Trends:
* **Training vs Validation Loss:** ![RoBERTa-base Training vs Validation Loss Across Epochs](images/RoBERTa-base_Training_vs_Validation_Loss_Across_Epochs.png)
* **Validation Accuracy & F1-Score Trend:** ![RoBERTa-base Validation Accuracy & F1-Score Trend](images/RoBERTa-base_Validation_Accuracy_&_F1-Score_Trend.png)

### 4. Ultimate Architecture Comparison 
A direct visual confrontation mapping the overall efficiency, metric balancing, and precision improvements across both algorithmic eras.
* ** Benchmark Analysis:** ![Transformer Models Performance Comparison](images/Transformer_Models_Performance_Comparison(di....png)

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/ShahriarKS/Cybercrime-Text-Classification.git](https://github.com/ShahriarKS/Cybercrime-Text-Classification.git)

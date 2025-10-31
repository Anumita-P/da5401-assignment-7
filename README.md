# DA5401 A7 — Multi-Class Model Selection using ROC and PR Curves  
**Name:** Anumita  
**Roll No:** BE22B004  

---

## Objective  
This assignment focuses on **multi-class model selection** using the **Landsat Satellite dataset**.  
We evaluate and compare the performance of several classifiers using **ROC-AUC** and **Precision–Recall (PRC)** curves to determine the best and worst models.  
The goal is to interpret model performance across thresholds rather than rely solely on accuracy.

---

## Dataset  
**Source:** [UCI Machine Learning Repository – Statlog (Landsat Satellite)](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)  
**Description:**  
- Multi-spectral pixel values (3×3 neighborhoods) in satellite imagery  
- 36 features, 6435 samples  
- Target: 6 land cover classes (Class 6 missing in dataset)  
- No missing values  

---

## Steps & Methodology  

### **1. Data Preparation**
- Loaded dataset using `ucimlrepo.fetch_ucirepo()`
- Encoded categorical class labels numerically  
- Standardized features using `StandardScaler`  
- Split into **70% training** and **30% testing** data  

### **2. Models Trained**
| Model | Library | Expected Performance |
|--------|----------|----------------------|
| Logistic Regression | `sklearn.linear_model` | Good |
| Random Forest | `sklearn.ensemble` | Excellent |
| SVM (One-vs-Rest) | `sklearn.svm` | Good |
| K-Nearest Neighbors | `sklearn.neighbors` | Good |
| Dummy Classifier | `sklearn.dummy` | Poor (Baseline) |
| Gaussian Naive Bayes | `sklearn.naive_bayes` | Moderate |
| Neural Network (MLP) | `sklearn.neural_network` | Excellent |
| XGBoost | `xgboost` | Excellent |
| Flipped Logistic Regression | Custom (for AUC < 0.5 demo) | Worst (AUC ≈ 0.04) |

---

## Evaluation Metrics  
- **Accuracy** — Ratio of correctly predicted samples  
- **Weighted F1-Score** — Handles class imbalance  
- **ROC-AUC (One-vs-Rest)** — Measures separability  
- **PRC-AUC (One-vs-Rest)** — Focuses on positive class precision-recall tradeoff  

---

## Results Summary  

| Model | Accuracy | Weighted F1 | Macro ROC-AUC | Macro PRC-AUC |
|--------|-----------|--------------|----------------|----------------|
| Logistic Regression | 0.838 | 0.812 | 0.96 | 0.89 |
| Random Forest | **0.917** | **0.914** | **0.99** | **0.94** |
| SVM | 0.897 | 0.896 | 0.97 | 0.92 |
| KNN | 0.911 | 0.910 | 0.98 | 0.94 |
| Dummy Classifier | 0.238 | 0.092 | 0.50 | 0.58 |
| Gaussian NB | 0.793 | 0.800 | 0.91 | 0.85 |
| Neural Network | 0.900 | 0.898 | 0.99 | 0.94 |
| XGBoost | **0.917** | **0.916** | **0.99** | 0.93 |
| Flipped Logistic Regression | 0.0005 | 0.0002 | **0.04** | 0.05 |

---

## Key Observations  

- **Best Models:** Random Forest, XGBoost, and Neural Network  
  - ROC-AUC ≈ 0.99, PRC-AUC ≈ 0.94  
  - Strong precision-recall balance and separability  

- **Worst Model:** Flipped Logistic Regression  
  - AUC = 0.04 → performs **worse than random chance**, meaning it systematically predicts incorrect classes  

- **Dummy Classifier:** Baseline with no discriminative ability (AUC ≈ 0.5)  

- **Trade-offs:**  
  Logistic Regression and GaussianNB had decent ROC-AUC but lower PRC-AUC due to more false positives when recall increased.

---

## Final Recommendation  
The **Random Forest Classifier** is the most reliable model.  
It maintains high precision and recall across thresholds, performs robustly across all classes, and achieves top scores in both ROC and PRC metrics.  
Neural Network and KNN also perform competitively but require more computation.

---

##  Bonus Exploration  
- **XGBoost** performed equally well as Random Forest (macro-AUC = 0.99).  
- **Flipped Logistic Regression** was intentionally inverted to visualize a poor-performing model (AUC < 0.5).  

---

##  Libraries Used  
python
pandas, numpy, sklearn, xgboost, matplotlib, ucimlrepo

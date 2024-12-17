# Fraud Transaction Detection

This project aims to detect fraudulent transactions using a robust Artificial Neural Network (ANN) model. The model was trained efficiently with extensive data preprocessing and feature engineering, achieving an accuracy of **99.91%**. The training process was optimized to complete in **under 1 minute** due to streamlined workflows.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [How to Run](#how-to-run)
9. [Further Improvements](#further-improvements)
10. [References](#references)

---

## Overview

Fraudulent transactions are a growing concern in financial systems. This project tackles the challenge by:
- Cleaning and preparing the transaction dataset.
- Identifying and removing irrelevant features.
- Using ANN to classify transactions as fraud or not fraud.

The solution optimizes accuracy while reducing processing time and computational complexity.

---

## Dataset

The dataset includes **83561 transactions** with 11 features:
- `step`: Step number in the simulation.
- `type`: Type of transaction (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
- `amount`: Amount transferred in the transaction.
- `nameOrig` & `nameDest`: Origin and destination account names.
- `oldbalanceOrg` & `newbalanceOrg`: Balance details before and after the transaction for the origin account.
- `oldbalanceDest` & `newbalanceDest`: Balance details for the destination account.
- `isFraud`: Target column indicating whether the transaction is fraud.
- `isFlaggedFraud`: Column indicating flagged fraudulent transactions.

---

## Data Preprocessing

1. **Null Value Removal**: Dropped rows with missing values.
2. **Feature Removal**:
   - Removed `nameOrig`, `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`, and `nameDest` for redundancy or lack of predictive power.
3. **Categorical Encoding**:
   - Created dummy variables for `type` (CASH_OUT, TRANSFER) after removing `CASH_IN`, `DEBIT`, and `PAYMENT`.
4. **Outlier Removal**:
   - Outliers were removed from `fraud_data` only to maintain fraud-specific patterns.

---

## Feature Engineering

- Removed **CASH_IN**, **DEBIT**, and **PAYMENT** transactions as no fraud was detected in these categories.
- Encoded transaction types using one-hot encoding for better model performance.

---

## Model Training

The **Artificial Neural Network** architecture:
1. **Input Layer**:
   - 64 neurons with `ReLU` activation.
2. **Hidden Layers**:
   - 32 neurons with `ReLU` activation.
3. **Output Layer**:
   - 1 neuron with `Sigmoid` activation for binary classification.

### Training Specifications:
- Optimizer: `Adam`
- Loss Function: `Binary Crossentropy`
- Epochs: 32
- Batch Size: 32
- Train-Test Split: 80-20

---

## Results

- **Accuracy**: 99.91%
- **Precision**: ~High
- **Recall**: ~High
- **F1-Score**: ~High

### Example Metrics:
- The ANN effectively detects fraudulent transactions with minimal false positives and false negatives.
- Classification report and performance metrics are evaluated for the test dataset.

---

## Conclusion

The project successfully predicts fraudulent transactions using an ANN. Key takeaways:
- Data preprocessing and feature engineering significantly improve efficiency and accuracy.
- Fraudulent transaction patterns were identified through thorough visualizations and boxplots.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/abhyudyasangwan/fraud_detection
   cd fraud-transaction-detection
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
   ```

3. Run the script:
   ```bash
   python fraud_detection.py
   ```

---

## Further Improvements

1. Implement anomaly detection techniques for unsupervised fraud detection.
2. Integrate advanced models like XGBoost or ensemble methods for comparison.
3. Deploy the model using Flask for real-time fraud detection.

---

## References

1. Original dataset: [Kaggle/Online Resource](#)
2. Libraries: TensorFlow, Pandas, Matplotlib, Seaborn, Scikit-learn.


# Betabank Customer Churn Analysis

## Overview

This project presents a comprehensive analysis and predictive modeling workflow for customer churn at Betabank. Using real-world banking data, the notebook explores, preprocesses, and models customer behavior to identify those most likely to leave the bank. The goal is to help Betabank improve customer retention and reduce churn-related losses.

## Features
- **Data Exploration:** In-depth analysis of customer demographics, account activity, and churn patterns.
- **Preprocessing:** Handling missing values, encoding categorical variables, and feature scaling.
- **Class Imbalance Solutions:** Techniques such as upsampling and downsampling to address imbalanced data.
- **Modeling:** Implementation and comparison of machine learning models including Random Forest and Logistic Regression.
- **Evaluation:** Use of F1 score and AUC-ROC metrics to assess model performance.
- **Interpretability:** Insights and recommendations for actionable business strategies.

## How to Use
1. Clone this repository or download the notebook.
2. Ensure you have the required dependencies (see below).
3. Open `betabank_analysis.ipynb` in Jupyter Notebook or VS Code.
4. Run the cells sequentially to reproduce the analysis and results.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Data
The analysis uses the `Churn.csv` dataset, which contains anonymized customer information, account activity, and churn status. The data is located in the `data/` directory.

## Project Structure
```
betabank_case/
├── betabank_analysis.ipynb
├── data/
│   └── Churn.csv
└── README.md
```

## Results
- Achieved high predictive performance using Random Forest with class balancing techniques.
- Provided actionable insights for Betabank to target at-risk customers and improve retention.



---



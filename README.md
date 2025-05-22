# Vehicle Classification with Supervised and Unsupervised Learning

## Project Summary

This project explores both **supervised** and **unsupervised** machine learning techniques to classify vehicles (Bus, Car, Van) based on geometric features extracted from silhouette images. It was developed for Prospect Auto, a car repair chain aiming to automate vehicle recognition.

---

## Notebooks Overview

### 1. EDA
- Explores class distributions and feature relationships
- Identifies overlap in feature space that justifies ML modeling

### 2. Preprocessing
- Cleans and scales the dataset
- Splits features/targets for supervised models
- Applies PCA for unsupervised analysis

### 3. Supervised Classification
- Models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
- Best Performers: Logistic Regression and SVM (~99.4% accuracy)
- Metrics: Accuracy, Confusion Matrix, Classification Report

### 4. Unsupervised Clustering
- Techniques: PCA, K-Means, DBSCAN
- K-Means captured 3 clusters aligned with classes (Silhouette Score: 0.30)
- DBSCAN struggled due to feature distribution (Score: 0.08)

---

## Tech Stack

- Python
- Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn

---

## Files & Outputs

- `processed_X_train.csv`: Scaled training features  
- `processed_X_test.csv`: Scaled test features  
- `processed_y_train.csv`: Training labels  
- `processed_y_test.csv`: Test labels  
- `processed_features_pca.csv`: PCA-reduced dataset for clustering

---

## Key Takeaways

- Simpler models (Logistic Regression, SVM) outperformed complex ones
- K-Means provided insightful clustering, even without labels
- The Bus class proved most challenging to classify

---

## Author

Developed by Dido De Boodt | Project for Masterschool


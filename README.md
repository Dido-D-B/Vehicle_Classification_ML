# Prospect Auto - Vehicle Classification (Supervised and Unsupervised Learning)

End-to-end machine learning project for classifying vehicle types (bus, car, van) from **geometric silhouette features**. You’ll find both supervised models (LogReg, SVM, Random Forest, XGBoost) and unsupervised clustering (PCA + K-Means, DBSCAN), with clear notebooks and a concise PDF report.

**Dataset**

* 18 numerical features extracted from vehicle silhouettes
* 3 classes: Bus, Car, Van
* Moderate class imbalance; PCA used for clustering prep. 

## Project Structure

```
├── Data/
│   ├── vehicle_raw.csv
│   ├── processed_X_train.csv
│   ├── processed_X_test.csv
│   ├── processed_y_train.csv
│   ├── processed_y_test.csv
│   ├── processed_features_scaled.csv
│   └── processed_features_pca.csv
├── Notebooks/
│   ├── EDA_Prospect_Auto.ipynb
│   ├── Preprocessing_Prospect_Auto.ipynb
│   ├── Supervised_Classification_Prospect_Auto.ipynb
│   └── Unsupervised_Clustering_Prospect_Auto.ipynb
├── Detailed_Report.pdf
└── README.md
```

## Project Overview

### EDA

- Explores class distributions and feature relationships
- Identifies overlap in feature space that justifies ML modeling

### Preprocessing
- Cleans and scales the dataset
- Splits features/targets for supervised models
- Applies PCA for unsupervised analysis

### Supervised Classification
- Models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
- Best Performers: Logistic Regression and SVM (~99.4% accuracy)
- Metrics: Accuracy, Confusion Matrix, Classification Report

### Unsupervised Clustering
- Techniques: PCA, K-Means, DBSCAN
- K-Means captured 3 clusters aligned with classes (Silhouette Score: 0.30)
- DBSCAN struggled due to feature distribution (Score: 0.08)

## Results

- **Supervised**: Logistic Regression and SVM reached ~99.4% accuracy on test data; simple linear models beat more complex trees here.  ￼
- **Unsupervised**: K-Means uncovered meaningful structure (silhouette ≈ 0.30) while DBSCAN performed poorly (silhouette ≈ 0.08) on these uniform features.  ￼

## Key Takeaways

- Simpler models (Logistic Regression, SVM) outperformed complex ones
- K-Means provided insightful clustering, even without labels
- The Bus class proved most challenging to classify

## Author

Developed by Dido De Boodt | Project for Masterschool


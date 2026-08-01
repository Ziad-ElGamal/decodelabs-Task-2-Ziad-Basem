# Project 2: Data Classification Using AI

## Overview
This repository contains an advanced predictive classification model built during the DecodeLabs industrial training program[cite: 1]. The objective of this project is to implement Supervised Learning to categorize data without relying on hard-coded heuristic rules[cite: 1]. 

## The Architecture: IPO Framework
The model was constructed using the Input-Process-Output (IPO) framework to ensure structural integrity[cite: 1]:

* **Input:** Utilizes the benchmark Iris dataset, containing 150 balanced samples across 3 classes and 4 dimensions (Sepal/Petal Length and Width)[cite: 1].
* **Process:** The data is randomized and split into an 80% training set and a 20% testing set[cite: 1]. It is scaled using `StandardScaler` to ensure equal feature weighting. The core logic is powered by the K-Nearest Neighbors (KNN) algorithm[cite: 1]. 
* **Output:** Validation is handled using a Confusion Matrix to track True/False Positives and Negatives[cite: 1]. The model's success is measured via the F1 Score to find the harmonic mean between Precision and Recall[cite: 1].

## Advanced Enhancements (Top Performer Upgrade)
To ensure a robust, professional-grade pipeline, this project goes beyond the baseline requirements by implementing:
* **Hyperparameter Tuning:** Utilized `GridSearchCV` with 5-fold cross-validation to programmatically discover the absolute optimal hyperparameters (K-value, distance metric, and weights) instead of relying on a hardcoded assumption.
* **Comprehensive Validation:** Generated a detailed classification report breaking down Precision, Recall, and F1-Score for each specific target class.
* **Data Visualization:** Integrated `seaborn` and `matplotlib` to dynamically plot and export a color-coded heatmap of the Confusion Matrix for better visual analysis.

## Technologies Used
* Python
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Author
**Ziad Basem El Gamal**
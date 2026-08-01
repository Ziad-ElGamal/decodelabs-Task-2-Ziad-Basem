import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("--- DecodeLabs Project 2: Advanced KNN Classification ---")
    
    # 1. INPUT: Load Data
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    flower_names = iris.target_names

    # 2. PROCESS: Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. ADVANCED UPGRADE: Hyperparameter Tuning via GridSearchCV
    print("Running GridSearchCV to find optimal hyperparameters...")
    param_grid = {
        'n_neighbors': range(1, 15),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # GridSearch systematically tests every combination of the parameters above
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn = grid_search.best_estimator_
    print(f"Optimal Parameters Found: {grid_search.best_params_}")

    # 4. PREDICT
    predictions = best_knn.predict(X_test_scaled)

    # 5. OUTPUT: Advanced Metrics & Visualizations
    print("\n--- Model Evaluation ---")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=flower_names))

    # Generate Confusion Matrix Heatmap
    conf_matrix = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
                xticklabels=flower_names, yticklabels=flower_names)
    plt.title('KNN Classification - Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save the figure automatically for your portfolio
    plt.savefig('confusion_matrix_heatmap.png')
    print("\nSuccess: 'confusion_matrix_heatmap.png' has been saved to your directory.")
    plt.show()

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

# ==========================================
# 1. INPUT
# ==========================================
# Load the 150-sample Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# ==========================================
# 2. PROCESS
# ==========================================
# Shuffle and split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to Mean=0 and Variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the KNN model with K=5 (Proven optimal)
model = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training set 
model.fit(X_train_scaled, y_train)

# Predict outcomes using the testing set
predictions = model.predict(X_test_scaled)

# ==========================================
# 3. OUTPUT
# ==========================================
# Validate using the Confusion Matrix and F1 Score
conf_matrix = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print("Confusion Matrix:\n", conf_matrix)
print("\nF1 Score:", round(f1, 4))
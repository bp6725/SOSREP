import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SOSREP.src.sosrep import SOSREP

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the SOSREP model
sosrep = SOSREP()
sosrep.fit(X_train_scaled, X_test_scaled, kernel_type='SDO', Bs=np.logspace(-2, 2, 10), n_iters_optim=500)

# Make predictions
train_predictions = sosrep.predict(X_train_scaled)
test_predictions = sosrep.predict(X_test_scaled)

# Print some results
print(f"Optimal b: {sosrep.optimal_b}")
print(f"Shape of train predictions: {train_predictions.shape}")
print(f"Shape of test predictions: {test_predictions.shape}")


# Optional: Visualize results for the first two features
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=train_predictions.flatten().cpu(), cmap='viridis')
plt.colorbar(label='Squared Density Estimate')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('SOSREP Density Estimation on Iris Dataset (Train)')
plt.show()
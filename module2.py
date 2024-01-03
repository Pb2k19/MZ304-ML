import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data for the example
np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
X = X + 2 * np.random.randn(200, 1)  # Introduce some non-linearity

# Create a DataFrame for better visualization
data = pd.DataFrame({"Feature": X.flatten(), "Label": y.flatten()})

# Visualize the synthetic data
plt.scatter(X, y, color="coral", label="Synthetic Data")
plt.title("Synthetic Data for Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = model_lr.predict(X_test)

# Evaluate the linear regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr:.2f}")
print(f"Linear Regression R^2 Score: {r2_lr:.2f}")

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)

y_pred_poly = model_poly.predict(X_poly_test)

# Evaluate the polynomial regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression MSE: {mse_poly:.2f}")
print(f"Polynomial Regression R^2 Score: {r2_poly:.2f}")

# Ridge Regression (Regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_scaled, y)

# Make predictions on the test set
y_pred_ridge = model_ridge.predict(X_scaled)

# Evaluate the Ridge regression model
mse_ridge = mean_squared_error(y, y_pred_ridge)
r2_ridge = r2_score(y, y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge:.2f}")
print(f"Ridge Regression R^2 Score: {r2_ridge:.2f}")

# Cross-Validation Score for Ridge Regression
cv_scores = cross_val_score(
    Ridge(alpha=1.0), X_scaled, y, cv=5, scoring="neg_mean_squared_error"
)
mean_cv_mse = -np.mean(cv_scores)
print(f"Mean Cross-Validation MSE for Ridge Regression: {mean_cv_mse:.2f}")

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, color="coral", label="Actual Data")
plt.scatter(X_test, y_pred_lr, color="blue", label="Linear Regression Prediction")
plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, color="coral", label="Actual Data")
plt.scatter(
    X_test, y_pred_poly, color="green", label="Polynomial Regression Prediction"
)
plt.title("Polynomial Regression")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X, y, color="coral", label="Actual Data")
plt.scatter(X, y_pred_ridge, color="purple", label="Ridge Regression Prediction")
plt.title("Ridge Regression")
plt.xlabel("Feature")
plt.ylabel("Label")
plt.legend()

plt.tight_layout()
plt.show()

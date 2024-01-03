import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate a hypothetical sales dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
sales_data = pd.DataFrame({"Sales": y, "Advertising Spend": X.flatten()})

# Simulate missing values
sales_data.loc[sales_data.sample(frac=0.1).index, "Sales"] = np.nan

# Data Cleaning: Impute missing values with mean
sales_data["Sales"].fillna(sales_data["Sales"].mean(), inplace=True)

# Feature Engineering: Create a new feature 'ROI' (Return on Investment)
sales_data["ROI"] = sales_data["Sales"] / sales_data["Advertising Spend"]

# Visualize the distribution of Sales and Advertising Spend
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(sales_data["Sales"], kde=True, color="skyblue")
plt.title("Distribution of Sales")

plt.subplot(1, 2, 2)
sns.histplot(sales_data["Advertising Spend"], kde=True, color="salmon")
plt.title("Distribution of Advertising Spend")

plt.tight_layout()
plt.show()

# Scatter plot of Advertising Spend vs. Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Advertising Spend", y="Sales", data=sales_data, color="coral")
plt.title("Scatter Plot of Advertising Spend vs. Sales")
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.show()

# Linear Regression to predict Sales based on Advertising Spend using NumPy
X_train, X_test, y_train, y_test = train_test_split(
    sales_data[["Advertising Spend"]],
    sales_data["Sales"],
    test_size=0.2,
    random_state=42,
)

# Adding a column of ones to X_train for the intercept term
X_train_np = np.c_[np.ones(X_train.shape[0]), X_train.to_numpy()]

# Linear regression using normal equation (closed-form solution)
theta = np.linalg.inv(X_train_np.T @ X_train_np) @ X_train_np.T @ y_train.to_numpy()

# Predictions on test set
X_test_np = np.c_[np.ones(X_test.shape[0]), X_test.to_numpy()]
y_pred = X_test_np @ theta

# Evaluate the model
mse = np.mean((y_test.to_numpy() - y_pred) ** 2)
print(f"\nMean Squared Error: {mse:.2f}")

# Visualize the linear regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Advertising Spend",
    y="Sales",
    data=sales_data,
    color="coral",
    label="Actual Sales",
)
sns.lineplot(
    x=X_test["Advertising Spend"], y=y_pred, color="blue", label="Regression Line"
)
plt.title("Linear Regression: Advertising Spend vs. Sales (NumPy)")
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data for binary classification
X, y = make_classification(
    n_samples=400,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42,
)

# Create a DataFrame for better visualization
data = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(1, 6)])
data["Label"] = y

# Visualize the synthetic data
plt.scatter(
    data["Feature 1"],
    data["Feature 2"],
    c=data["Label"],
    cmap="coolwarm",
    edgecolors="k",
    marker="o",
)
plt.title("Binary Classification Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Label")
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Data Preprocessing: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Binary Classification Models
model_lr = LogisticRegression(max_iter=10000)
model_svm = SVC(probability=True)

# Hyperparameter tuning using GridSearchCV
param_grid_lr = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

grid_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, scoring="accuracy")
grid_svm = GridSearchCV(model_svm, param_grid_svm, cv=5, scoring="accuracy")

grid_lr.fit(X_train_scaled, y_train)
grid_svm.fit(X_train_scaled, y_train)

best_lr = grid_lr.best_estimator_
best_svm = grid_svm.best_estimator_

# Feature Importance (Random Forest)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)

importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]


# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(y_test, y_pred))


evaluate_model(best_lr, X_test_scaled, y_test)
evaluate_model(best_svm, X_test_scaled, y_test)


# ROC Curve
def plot_roc_curve(model, X_test, y_test, title):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve(best_lr, X_test_scaled, y_test, "Logistic Regression ROC Curve")
plot_roc_curve(best_svm, X_test_scaled, y_test, "Support Vector Machine ROC Curve")

# Visualization of feature importances
plt.figure()
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), data.columns[:-1][indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()

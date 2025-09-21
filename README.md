import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Generation and Preprocessing ---
# Generate a synthetic dataset to simulate credit risk data.
np.random.seed(42)
data_size = 1000
data = {
    'income': np.random.normal(50000, 15000, data_size),
    'debt': np.random.normal(10000, 5000, data_size),
    'payment_history': np.random.randint(1, 10, data_size),
    'default': np.random.choice([0, 1], data_size, p=[0.8, 0.2])
}
df = pd.DataFrame(data)

# Create a more realistic relationship between features and default
df.loc[df['income'] < 30000, 'default'] = 1
df.loc[df['debt'] > 20000, 'default'] = 1
df.loc[df['payment_history'] < 4, 'default'] = 1

# --- 2. Feature Engineering ---
# Create new, meaningful features from existing data.
df['debt_to_income_ratio'] = df['debt'] / df['income']
df['payment_score'] = df['payment_history'] * 10

# --- 3. Data Splitting and Scaling ---
# Define features (X) and target (y)
features = ['income', 'debt', 'payment_history', 'debt_to_income_ratio', 'payment_score']
X = df[features]
y = df['default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for algorithms sensitive to feature magnitude.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Model Training and Evaluation ---
# Train and evaluate both Logistic Regression and Random Forest models.

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]

print("--- Logistic Regression Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log_reg):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_log_reg):.2f}")

print("\n" + "="*50 + "\n")

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)
y_prob_rf = rf_clf.predict_proba(X_test_scaled)[:, 1]

print("--- Random Forest Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.2f}")

print("\n" + "="*50 + "\n")

# --- 5. Visualization of Results ---

# ROC Curve
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(10, 7))
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_log_reg):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Credit Scoring Models')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

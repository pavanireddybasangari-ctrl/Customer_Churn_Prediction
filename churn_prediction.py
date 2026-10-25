import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("customer_churn.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# 2. Convert TotalCharges to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 3. Remove missing values
df = df.dropna(subset=["TotalCharges"])

# 4. Remove customer ID
df = df.drop("customerID", axis=1)

# 5. Convert Churn into 0 and 1
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 6. Convert all text columns into numbers
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include=["object"]).columns,
    drop_first=True,
    dtype=int
)

# 7. Separate input and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 8. Check for remaining text columns
print("Text columns remaining:")
print(X.select_dtypes(include=["object"]).columns)

# 9. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 10. Create the model
model = LogisticRegression(max_iter=1000)

# 11. Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# 12. Make predictions
y_pred = model.predict(X_test)

# 13. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# 14. Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 15. Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Customer Churn Prediction completed!")
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Accuracy Percentage:", rf_accuracy * 100, "%")
print("\n===== MODEL COMPARISON =====")

print("Logistic Regression Accuracy:",
      accuracy * 100, "%")

print("Random Forest Accuracy:",
      rf_accuracy * 100, "%")

if rf_accuracy > accuracy:
    print("Better Model: Random Forest")
else:
    print("Better Model: Logistic Regression")
    import joblib

# Save the best model
joblib.dump(model, "churn_model.pkl")

print("\nBest model saved successfully!")
print("Customer Churn Prediction completed!")

# Save the best model
import joblib

joblib.dump(model, "churn_model.pkl")

print("Best model saved successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("feature.csv")

# Remove non-existent domains before training
df = df[df['validDomain'] == 1]
df = df.drop(columns=['validDomain'])

# Drop 'Domain' column if it exists
if 'Domain' in df.columns:
    df = df.drop(columns=['Domain'])

# Ensure 'label' column exists
if 'label' not in df.columns:
    raise ValueError("ERROR: 'label' column is missing in dataset! Ensure each row has a phishing (1) or legitimate (0) label.")

# Convert all features to numeric and handle missing values
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {name}")
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# Save the best model
joblib.dump(best_model, "best_phishing_model.pkl")
print(f"Best Model: {best_name} with Accuracy: {best_accuracy}")
print("Best model saved as 'best_phishing_model.pkl'.")

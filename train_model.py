
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Save feature names separately
feature_names = list(X.columns)
joblib.dump(feature_names, "features.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(), KNeighborsClassifier()
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "Support Vector Machine": make_pipeline(
        StandardScaler(), SVC(probability=True, random_state=42)
    )
}

best_name = ""
best_model = None
best_accuracy = -1

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(name, "Accuracy:", round(accuracy, 3))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name
        best_model = model

# Save the actual trained model
joblib.dump(best_model, "best_model.pkl")

print("Best model:", best_name)
print("Best accuracy:", round(best_accuracy, 3))
print("Model and feature names saved successfully.")

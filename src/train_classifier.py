import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("../data/processed_data.csv")
X = df[["website_category", "time_spent", "frequency", "day_type"]]
y = df["productivity_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
with open("../models/saved_models.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../results/evaluation_metrics.txt", "w") as f:
    f.write(f"Classifier Accuracy: {accuracy}\n\n")
    f.write(report)
print("Classifier trained successfully.")
print("Accuracy:", accuracy)

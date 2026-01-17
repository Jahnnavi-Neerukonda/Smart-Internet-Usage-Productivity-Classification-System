import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Load processed data
df = pd.read_csv("../data/processed_data.csv")
X = df[["website_category", "time_spent", "frequency", "day_type"]]
y = df["productivity_score"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
with open("../results/evaluation_metrics.txt", "a") as f:
    f.write(f"\nRegression RMSE: {rmse}\n")

print("Regressor trained successfully.")
print("RMSE:", rmse)

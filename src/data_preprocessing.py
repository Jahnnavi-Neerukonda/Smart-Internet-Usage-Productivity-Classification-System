import pandas as pd
from sklearn.preprocessing import LabelEncoder
def preprocess_data():
    df = pd.read_csv("../data/raw_data.csv")
    le = LabelEncoder()
    df["website_category"] = le.fit_transform(df["website_category"])
    df["day_type"] = le.fit_transform(df["day_type"])
    df["productivity_label"] = le.fit_transform(df["productivity_label"])
    df.to_csv("../data/processed_data.csv", index=False)
    print("Data preprocessing completed and saved.")
if __name__ == "__main__":
    preprocess_data()

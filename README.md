# Smart Internet Usage & Productivity Classification System

An end-to-end machine learning system with an interactive web interface that predicts user productivity based on internet usage patterns.

---

## Problem Statement
Unstructured and excessive internet usage can negatively impact productivity.  
This project aims to analyze user behavior and classify whether their internet usage is **productive, neutral, or unproductive**, while also predicting a **productivity score (0–100)**.

---

## Key Features
- Productivity classification (Productive / Neutral / Unproductive)
- Productivity score prediction (0–100)
- Interactive web interface using Streamlit
- Feature importance analysis
- Confusion matrix visualization
- Model comparison for performance evaluation

---

## Project Pipeline
1. Data Collection (Simulated dataset)
2. Data Preprocessing (Label Encoding)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Comparison
7. Model Evaluation
8. Prediction & Deployment (Streamlit UI)

---

## Dataset
The dataset includes the following features:
- Website category (Education, Social Media, Coding, etc.)
- Time spent (minutes)
- Frequency of visits
- Day type (Weekday / Weekend)
- Productivity label
- Productivity score

A structured synthetic dataset is used for experimentation and learning.

---

## Models Used

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier (Best Performance)

### Regression
- Random Forest Regressor

---

## Model Evaluation

### Classification Metrics
- Accuracy: ~85%
- Precision, Recall, F1-score
- Confusion Matrix (saved in `results/`)

### Regression Metrics
- RMSE: ~6.4

### Additional Analysis
- Feature Importance (visualized and saved)

---

## Results
- Random Forest outperformed other models in classification accuracy and stability.
- Feature importance analysis revealed key factors influencing productivity.
- The system successfully predicts both categorical and continuous productivity outputs.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## How to Run

```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/train_classifier.py
python src/train_regressor.py
streamlit run app.py

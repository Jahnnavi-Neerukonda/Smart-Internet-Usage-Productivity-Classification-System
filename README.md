# Smart Internet Usage & Productivity Classification System

A machine learning–based system that analyzes internet usage behavior and classifies it into **productive**, **neutral**, or **unproductive** categories.  
The project also predicts a **productivity score** to quantify digital behavior.

This project is designed as a structured AIML pipeline suitable for students and early-stage ML practitioners.

---

## Problem Statement
Excessive and unstructured internet usage negatively impacts productivity.  
This system helps analyze usage patterns and provides insights into how digital habits affect productivity.

---

## Project Pipeline
1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Model Training
6. Model Evaluation
7. Prediction

---

## Dataset
The dataset contains:
- Website category
- Time spent (minutes)
- Usage frequency
- Day type (Weekday / Weekend)
- Productivity label
- Productivity score

A simulated dataset is used for learning and experimentation purposes.

---

## Models Used
### Classification
- Random Forest Classifier

### Regression
- Random Forest Regressor

---

## Results
- Classification Accuracy: ~85%
- Regression RMSE: ~6.4

Detailed evaluation metrics are available in the `results/` directory.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## How to Run
```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/train_classifier.py
python src/train_regressor.py
python src/predict.py

import pickle
import numpy as np
with open("../models/saved_models.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input
# [website_category, time_spent, frequency, day_type]
sample_input = np.array([[1, 120, 4, 0]])

prediction = model.predict(sample_input)

print("Predicted Productivity Class:", prediction)

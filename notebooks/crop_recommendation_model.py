# Importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('Data/crop_recommendation.csv')

# Separate features and target label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Train the Random Forest model
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(features, target)

# Save the trained Random Forest model
with open('models/RandomForest.pkl', 'wb') as file:
    pickle.dump(RF, file)

# Define a function to make predictions
def predict_crop(data):
    """
    Predicts the crop based on input features.
    :param data: Input features in the format [N, P, K, temperature, humidity, ph, rainfall]
    :return: Predicted crop label
    """
    # Load the trained model
    with open('models/RandomForest.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    prediction = model.predict(data)
    return prediction

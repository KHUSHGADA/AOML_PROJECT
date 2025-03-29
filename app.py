import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")  # Ensure this file is in the same directory
    return df

df = load_data()

# Encode crop labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # Convert crop names to numbers

# Prepare training data
X = df.drop(columns=['label'])  # Features (N, P, K, Temp, Humidity, pH, Rainfall)
y = df['label']  # Target (Encoded crops)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("\U0001F33E Crop Recommendation System")
st.write("Enter the soil parameters to predict the best crop to grow!")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# Prediction
if st.button("Predict Crop"):
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(user_input)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]  # Convert number back to crop name
    
    st.success(f"\U0001F331 Recommended Crop: **{predicted_crop}**")

# Display model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ Model Accuracy: **{accuracy:.2f}**")

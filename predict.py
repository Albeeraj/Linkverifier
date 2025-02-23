import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_phishing_model.pkl")

# Function to take a new URL's extracted features and predict
def predict_url(features):
    df = pd.DataFrame([features])  # Convert input into DataFrame
    prediction = model.predict(df)[0]  # Get prediction
    return "Phishing" if prediction == 1 else "Legitimate"

# Example new URL features (Replace with real extracted features)
new_url_features = {
    "Have_IP": 1,
    "Have_At": 0,
    "URL_Length": 50,
    "URL_Depth": 3,
    "Redirection": 0,
    "https_Domain": 0,
    "TinyURL": 1,
    "Prefix/Suffix": 1,
    "DNS_Record": 0,
    "Web_Traffic": 1,
    "Domain_Age": 1,
    "Domain_End": 0,
    "iFrame": 0,
    "Mouse_Over": 1,
    "Right_Click": 0,
    "Web_Forwards": 0
}

# Predict if the URL is phishing or legitimate
result = predict_url(new_url_features)
print(f"Prediction: {result}")
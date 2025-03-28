import pandas as pd
import joblib
import socket

# Function to check if a domain exists
def is_valid_domain(domain):
    """Check if a domain has a valid DNS record."""
    try:
        socket.gethostbyname(domain)
        return 1  # Valid domain
    except socket.gaierror:
        return 0  # Invalid domain

# Load the trained model
model = joblib.load("best_phishing_model.pkl")

# Ask for user input
input_url = input("Enter a URL: ")

# Check if the domain is valid
if is_valid_domain(input_url) == 0:
    print("Invalid website: This domain does not exist.")
else:
    # Extract features (assuming function `extract_features` exists)
    features = extract_features(input_url)  
    features_df = pd.DataFrame([features])

    # Predict using the trained model
    prediction = model.predict(features_df)

    if prediction == 1:
        print("Phishing website detected!")
    else:
        print("Legitimate website.")

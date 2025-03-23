from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

app = FastAPI()
recent_urls = []  # Stores the last 10 checked URLs


# Load trained phishing detection model
try:
    model = joblib.load("best_phishing_model.pkl")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# Function to extract features from a URL
SAFE_DOMAINS = ["linkverifier-2.onrender.com"] 
def extract_features(url: str):
    features = {
        "Have_IP": 1 if "192." in url or "https://" not in url else 0,
        "Have_At": 1 if "@" in url else 0,
        "URL_Length": len(url),
        "URL_Depth": url.count('/'),
        "Redirection": 1 if "//" in url[7:] else 0,
        "https_Domain": 1 if url.startswith("https") else 0,
        "TinyURL": 1 if "bit.ly" in url or "tinyurl" in url else 0,
        "Prefix/Suffix": 1 if "-" in url else 0,
        "DNS_Record": 1,
        "Web_Traffic": 0,
        "Domain_Age": 1,
        "Domain_End": 0,
        "iFrame": 0,
        "Mouse_Over": 0,
        "Right_Click": 1,
        "Web_Forwards": 0
    }
    print(f"Extracted features for {url}: {features}")  # ✅ Debugging
    return features


# Prediction function with Safe List Check
def predict_url(url: str):
    # If the URL is in the safe list, return "Legitimate" immediately
    if any(safe in url for safe in SAFE_DOMAINS):
        return {"result": "Legitimate"}

    # Extract features and predict
    features = extract_features(url)
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return {"result": "Phishing" if prediction == 1 else "Legitimate"}


# Serve the frontend (index.html)
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index.html: {e}")

# API endpoint for prediction
import re  # Import Regular Expressions for URL validation
@app.post("/predict")
def predict(data: dict):
    url = data.get("url", "").strip()
    
    # ✅ URL Validation
    url_pattern = re.compile(
        r'^(https?:\/\/)?'  # http:// or https:// (optional)
        r'([\da-z\.-]+)\.([a-z\.]{2,6})'  # Domain name
        r'([\/\w \.-]*)*\/?$'  # Path (optional)
    )
    if not url_pattern.match(url):
        return {"result": "⚠️ Enter a valid URL, including path (e.g., https://example.com/path)"}

    # ✅ Perform phishing check
    result = predict_url(url)

    # ✅ Store only last 5 checked URLs (excluding invalid)
    if result["result"] != "⚠️ Enter a valid URL, including path (e.g., https://example.com/path)":
        recent_urls.insert(0, {"url": url, "result": result["result"]})
        if len(recent_urls) > 5:  # ✅ Now storing only the last 5 URLs
            recent_urls.pop()

    return result


# API endpoint to get the last 10 checked URLs
@app.get("/recent")
def get_recent():
    return {"recent_urls": recent_urls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

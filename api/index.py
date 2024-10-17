import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import scipy.sparse as sp
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import some helper functions and feature extraction 
from src.helpers import extract_urls
from src.featurize_url import UrlFeaturizer

# Get current directory
import os

cwd = os.getcwd()
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and TF-IDF vectorizer
model_path = os.path.join(cwd, "models/fraud_random_forest_model.pkl")
tfidf_vectorizer_path = os.path.join(cwd, "models/fraud_tfidf_vectorizer.pkl")
url_model_path = os.path.join(cwd, "models/url_random_forest_model.pkl")

# Load the trained model and TF-IDF vectorizer
model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
url_model = joblib.load(url_model_path)

# Define FastAPI instance
app = FastAPI()

# CORS settings
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Define a function to predict URL
def predict_url(url):
    featurizer = UrlFeaturizer(url)
    feature_vector = featurizer.extract_features()  # Hypothetical method to extract features
    prediction = url_model.predict([feature_vector])
    return int(prediction[0])

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Define Pydantic input model
class InputData(BaseModel):
    text: str

# Define endpoint for model inference
@app.post("/api/predict")
async def predict(data: InputData):
    try:
        url_prediction = 0 # Start off with not spam from url_model, unless there are links provided. 
        result = extract_urls(data.text)
        urls = result['url']
        text_message = result["text"]

        if len(urls) != 0:
            # We need to pass this into the urls model now 
            result = []
            for url_link in urls:
                result.append(predict_url(url_link))
            
            # Check if any prediction is for "spam"
            if any(label == 0 for label in result):
                url_prediction = 0 # Not spam 
            else:
                url_prediction = 1 # One type of spam (phishing, smishing, etc)

        # Preprocess the input text
        preprocessed_text = preprocess_text(text_message)
        
        # Vectorize the preprocessed text
        text_features = tfidf_vectorizer.transform([preprocessed_text])
        
        # Calculate message length
        message_length = len(preprocessed_text)
        
        # Concatenate TF-IDF features with message length
        features_with_length = np.hstack((text_features.toarray(), [[message_length]]))
        
        # Make predictions
        fraud_prediction = model.predict(features_with_length)

        # Convert prediction to standard Python data type (e.g., int)
        fraud_prediction = int(fraud_prediction[0])

        # Check both predictions 
        if fraud_prediction or url_prediction == 1:
            prediction = 1
        else:
            prediction = 0
        
        # Return the prediction
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
def test():
    return "User 1 Testing server endpoint!"

# For local testing 
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
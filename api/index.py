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

# Get current directory
import os

cwd = os.getcwd()
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and TF-IDF vectorizer
model_path = os.path.join(cwd, "models/fraud_random_forest_model.pkl")
tfidf_vectorizer_path = os.path.join(cwd, "models/fraud_tfidf_vectorizer.pkl")

# Load the trained model and TF-IDF vectorizer
model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

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
        # Preprocess the input text
        preprocessed_text = preprocess_text(data.text)
        
        # Vectorize the preprocessed text
        text_features = tfidf_vectorizer.transform([preprocessed_text])
        
        # Calculate message length
        message_length = len(preprocessed_text)
        
        # Concatenate TF-IDF features with message length
        features_with_length = np.hstack((text_features.toarray(), [[message_length]]))
        
        # Make predictions
        prediction = model.predict(features_with_length)

        # Convert prediction to standard Python data type (e.g., int)
        prediction = int(prediction[0])
        
        # Return the prediction
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
def test():
    return "Test server endpoint!"

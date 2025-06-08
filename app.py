# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from yelp_model import predict_sentiment

app = FastAPI()

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: ReviewInput):
    result = predict_sentiment(input_data.text)
    return result

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}

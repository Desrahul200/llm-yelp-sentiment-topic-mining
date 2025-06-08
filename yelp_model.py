# yelp_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Inference mode

def predict_sentiment(text: str):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Calculate probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1).tolist()[0]
    predicted_class = int(torch.argmax(outputs.logits, dim=-1).item())
    # Map numeric label to sentiment
    label_map = {0: "neutral", 1: "negative", 2: "positive"}
    sentiment = label_map[predicted_class]

    return {
        "sentiment": sentiment,
        "label": predicted_class,
        "probabilities": probabilities
    }

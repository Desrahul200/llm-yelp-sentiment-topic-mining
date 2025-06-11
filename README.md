# LLM-Powered Sentiment & Topic Mining – Yelp Reviews Analysis

This project leverages state-of-the-art Large Language Models (LLMs) and topic modeling techniques to analyze Yelp reviews, extracting both sentiment and key discussion topics. It provides a FastAPI-based web service for sentiment prediction and includes Jupyter notebooks for data preprocessing, model training (including DistilBERT fine-tuning), and topic mining.

---

## Features

- **Sentiment Analysis API**: Predicts sentiment (positive, negative, neutral) for Yelp reviews using a fine-tuned transformer model.
- **Topic Mining**: Uses BERTopic and sentence transformers to extract and visualize main topics from large review datasets.
- **Data Preprocessing**: Cleans and prepares raw Yelp review data for downstream NLP tasks.
- **Embeddings**: Utilizes BERT-based embeddings for both sentiment and topic modeling.

---

## Project Structure

```
.
├── app.py                # FastAPI app exposing sentiment prediction endpoint
├── yelp_model.py         # Model loading and sentiment prediction logic
├── requirements.txt      # Python dependencies
├── yelp_nlp_code.ipynb   # Data cleaning, preprocessing, modeling, and embedding extraction
├── yelp.ipynb            # Topic modeling and advanced analysis
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Yelp Open Dataset**  
   - Download the [Yelp Open Dataset](https://www.yelp.com/dataset) and place the review JSON file in your working directory or Google Drive as required by the notebooks.

4. **Download or place your fine-tuned sentiment model**  
   - The model should be placed in a directory named `sentiment_model` in the project root.

---

## Usage

### 1. Run the Sentiment Analysis API

```bash
uvicorn app:app --reload
```

- Visit `http://127.0.0.1:8000` to check the API status.
- Use the `/predict` endpoint (POST) with JSON:
  ```json
  {
    "text": "The food was amazing and the service was excellent!"
  }
  ```

### 2. Data Preprocessing, Modeling & Topic Mining

- Open and run `yelp_nlp_code.ipynb` for:
  - Loading and cleaning Yelp review data
  - Generating BERT embeddings
  - **Fine-tuning DistilBERT for sentiment classification**
  - Training and evaluating Neural Network, Logistic Regression, and Naive Bayes models

- Open and run `yelp.ipynb` for:
  - Topic modeling with BERTopic
  - Topic visualization and coherence analysis

---

## API Endpoints

- `GET /`  
  Health check endpoint.

- `POST /predict`  
  **Request Body:**  
  ```json
  { "text": "Your review text here" }
  ```
  **Response:**  
  ```json
  {
    "sentiment": "positive",
    "label": 2,
    "probabilities": [0.01, 0.05, 0.94]
  }
  ```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for Python packages:
  - fastapi
  - uvicorn
  - transformers
  - torch
  - pydantic

Additional packages (for notebooks):
- pandas, nltk, scikit-learn, bertopic, sentence-transformers, etc.

---

## Notebooks Overview

### **yelp_nlp_code.ipynb**
- **Dataset**:  
  Download the [Yelp Open Dataset](https://www.yelp.com/dataset) and place the review JSON file in your working directory or Google Drive.
- **Data Preprocessing**:  
  - Loads raw Yelp review data (JSON format).
  - Cleans and preprocesses text (tokenization, stopword removal, lemmatization).
  - Saves a cleaned CSV for fast access.
- **Embeddings**:  
  - Generates BERT embeddings for each review using a pre-trained BERT model.
- **Modeling & Fine-Tuning**:  
  - **DistilBERT Fine-Tuning**:  
    - Fine-tunes a `distilbert-base-uncased` transformer model for multi-class sentiment classification (positive, negative, neutral) using Hugging Face Transformers and the cleaned Yelp data.
    - Handles class imbalance, tokenization, and uses Trainer API for efficient training and evaluation.
    - Saves the fine-tuned model for downstream API inference.
  - **Neural Network Classifier**:  
    - Trains a simple feedforward neural network on BERT embeddings for sentiment classification.
    - Uses mixed-precision (FP16) and CUDA acceleration for efficiency.
    - Evaluates and saves the trained model.
  - **Logistic Regression & Naive Bayes**:  
    - Trains and evaluates both models using TF-IDF features extracted from the cleaned review text.
    - Provides classification reports and accuracy for both models.

### **yelp.ipynb**
- **Topic Modeling**:  
  - Uses BERTopic and sentence-transformers to extract and visualize main topics from a large sample of Yelp reviews.
  - Calculates topic coherence and visualizes topic clusters.
- **Additional Analysis**:  
  - Includes KMeans clustering on BERT embeddings for unsupervised topic discovery.

---

## Model

- The sentiment analysis model is a transformer-based sequence classifier (DistilBERT) fine-tuned on Yelp review data.
- The model and tokenizer are loaded from the `sentiment_model` directory.

---

## Example

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"The food was amazing and the service was excellent!\"}"
```

**Response:**
```json
{
  "sentiment": "positive",
  "label": 2,
  "probabilities": [0.01, 0.05, 0.94]
}
```

---

## Acknowledgements

- [Yelp Open Dataset](https://www.yelp.com/dataset)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BERTopic](https://maartengr.github.io/BERTopic/) 
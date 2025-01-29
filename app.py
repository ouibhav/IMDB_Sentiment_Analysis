from flask import Flask, request, jsonify
from flask import render_template
import joblib
from bs4 import BeautifulSoup
import re

# Loading the models and vectorizer
model = joblib.load('lr_tfidf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initializing Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Getting the review text from the request
    data = request.get_json()
    review_text = data.get('review_text', '')

    if not review_text:
        return jsonify({'error': 'No review_text provided'}), 400

    # Preprocessing the input (applying the same steps as during training)
    def preprocess_text(text):
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
        return text

    processed_text = preprocess_text(review_text)

    # Vectorize the input text
    text_vector = vectorizer.transform([processed_text])

    # Predicting sentiment
    sentiment_prediction = model.predict(text_vector)[0]

    # Converting prediction to a human-readable label
    sentiment_label = 'positive' if sentiment_prediction == 1 else 'negative'

    # Returning the prediction as JSON
    return jsonify({'sentiment_prediction': sentiment_label})


if __name__ == '__main__':
    app.run(debug=True)

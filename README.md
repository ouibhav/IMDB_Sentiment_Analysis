# IMDB Sentiment Analysis Pipeline

This project implements an end-to-end sentiment analysis pipeline using the IMDB movie reviews dataset. It includes data collection, storage, cleaning, model training, and a Flask web interface for user interaction.


## Features
- **Data Ingestion:** Loads IMDB movie reviews dataset.
- **Text Processing:** Cleans and preprocesses text data.
- **Model Training:** Uses TF-IDF vectorization and Logistic Regression.
- **Flask Web App:** Provides a user-friendly interface to analyze sentiment.

## Dataset

- **Source:** 
  IMDB Movie Reviews Dataset
- **Size:** 
  50,000 labeled reviews (positive/negative sentiment)
- **Access:** 
    - **Via Kaggle:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  


## Project Setup
 **1) Clone the repository:** 
  IMDB Movie Reviews Dataset
```bash
  git clone https://github.com/ouibhav/IMDB_Sentiment_Analysis.git 
```
**2) Create a virtual environment and activate it:** 
  IMDB Movie Reviews Dataset
  ```bash
  python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**3) Install dependencies:** 
  IMDB Movie Reviews Dataset
  ```bash
  pip install -r requirements.txt
```
## Model Training

**1) Run the Jupyter Notebook for data setup and model training:** 
```bash
  jupyter notebook data_setup_model_train.ipynb
```
**2) The trained model (lr_tfidf_model.pkl) and vectorizer (tfidf_vectorizer.pkl) will be saved for inference.**

## Running the Flask Web App

**1) Start the Flask server:** 
```bash
  python app.py
```
**2) Open your browser and go to http://127.0.0.1:5000/ to access the sentiment analysis UI.**

## Web Interface Usage

**1) Enter a movie review in the text box.** 


**2) Click the Predict button.**

**3) View the predicted sentiment (positive or negative).**
![Image](https://github.com/user-attachments/assets/e6b7947e-493d-4966-a0c5-e0bcd8a9d5b8)
## Exploratory Data Analysis (EDA)

**1) Number of reviews per sentiment (distribution)** 
```bash
sentiment
positive    25000
negative    25000
Name: count, dtype: int64
```
![Image](https://github.com/user-attachments/assets/9439f2e7-9090-417c-be60-09a857db866b)

**2) Average review length for positive vs. negative**
```bash
sentiment
negative    229.46456
positive    232.84932
Name: review_length, dtype: float64
```
![Image](https://github.com/user-attachments/assets/73563f22-ad50-4041-9178-1ad89e41c82f)
## Model Information

**1) Vectorization:**  TF-IDF 

**2) Model:** Logistic Regression
  - Accuracy Bag of Words: **0.8599**
  - Accuracy TF-IDF: **0.8867**
  - ```bash
               precision    recall  f1-score   support

    Positive       0.89      0.88      0.89      4993
    Negative       0.88      0.89      0.89      5007

    accuracy                           0.89     10000
    macro avg      0.89      0.89      0.89     10000
    weighted avg   0.89      0.89      0.89     10000
    ```


## Conclusion

**The analysis of the IMDB movie reviews dataset reveals key insights into sentiment distribution, review length, and model performance:**

**1) Sentiment Distribution:** The dataset has a fairly balanced number of positive and negative reviews, ensuring that our sentiment analysis model does not lean toward one class.

**2) Review Length Analysis:** On average, positive reviews tend to be longer than negative ones, indicating that users who enjoyed a movie often provide more detailed feedback.

**3) Model Performance:** We observed that both Logistic Regression and Multinomial Naïve Bayes models perform well compared to Linear Support Vector Machines (SVM). However, there is still room for improvement.

**4) Potential Enhancements:** The accuracy of the models can be further improved by enhancing data preprocessing techniques and incorporating lexicon-based models like TextBlob, which leverage sentiment dictionaries for better classification.

**Moving forward, refining feature engineering, experimenting with ensemble models, and integrating deep learning approaches could further boost sentiment classification accuracy.**

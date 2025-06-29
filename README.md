# ⚽ FIFA Tweet Sentiment Analyzer

This project is a simple Machine Learning web app that predicts the **sentiment of tweets related to the FIFA World Cup** (positive, neutral, or negative). Built using **Gradio**, **Scikit-learn**, and **NLTK**, this project was developed as part of an internship project on Sentiment Analysis.

![Demo](https://huggingface.co/spaces/ansalcm/fifa-tweet-sentiment-analyzer/resolve/main/demo.gif) <!-- optional if you add a GIF or screenshot -->

## 🚀 Live App

👉 [Try the Sentiment Analyzer on Hugging Face](https://huggingface.co/spaces/ansalcm/fifa-tweet-sentiment-analyzer)

## 🧾 Project Notebook

👉 [View full Google Colab notebook](https://colab.research.google.com/drive/1PygmPwNJTBMOMWylOPpUAR7VEbRFnIg3)

The notebook includes data exploration, preprocessing, model training, evaluation, and saving steps.

## 📦 Technologies Used

- Python
- Scikit-learn
- NLTK
- Gradio
- Jupyter/Colab (for model training)
- Hugging Face Spaces (for deployment)

## 📁 Project Structure

├── app.py # Gradio web interface
├── requirements.txt # Dependencies for Hugging Face
├── logistic_regression.pkl # Trained sentiment classification model
└── tfidf_vectorizer.pkl # TF-IDF vectorizer used during training

markdown
Copy
Edit

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Input**: Preprocessed tweet text
- **Output**: Predicted sentiment (`positive`, `neutral`, or `negative`)
- **Vectorizer**: TF-IDF

## 🔄 How It Works

1. Tweet text is cleaned and preprocessed (lowercased, stopwords removed, lemmatized).
2. The text is vectorized using the TF-IDF model.
3. The trained model predicts the sentiment.
4. A confidence score is returned along with the prediction.

## 🛠 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fifa-tweet-sentiment-analyzer.git
   cd fifa-tweet-sentiment-analyzer
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
📚 NLTK Downloads Used
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
👨‍💻 Author
Ansal CM
Internship Project on Sentiment Analysis
Deployed using Hugging Face Spaces



# ⚽ FIFA Tweet Sentiment Analyzer

This project is a beginner-friendly **Machine Learning app** that predicts the **sentiment of tweets related to the FIFA World Cup** — using Natural Language Processing (NLP), TF-IDF, and Logistic Regression. The app is built using **Gradio** and deployed on **Hugging Face Spaces**.

## 🚀 Live Demo

👉 [🌐 Try the App on Hugging Face](https://huggingface.co/spaces/ansalcm/fifa-tweet-sentiment-analyzer)

## 🧾 Project Notebook

📘 You can view the full Colab notebook, which includes:
- Data exploration
- Text preprocessing
- Model training
- Evaluation
- Saving models for deployment

🔗 [Open in Google Colab](https://colab.research.google.com/drive/1PygmPwNJTBMOMWylOPpUAR7VEbRFnIg3?usp=sharing)

## 📦 Technologies Used

- Python 3
- Scikit-learn
- NLTK
- Gradio
- Google Colab
- Hugging Face Spaces

## 🧠 How It Works

1. The user inputs a tweet related to the FIFA World Cup.
2. The text is preprocessed (cleaned, tokenized, lemmatized).
3. The text is transformed using a pre-trained TF-IDF vectorizer.
4. The Logistic Regression model predicts the sentiment (positive, neutral, or negative).
5. The result and model confidence are displayed in a Gradio UI.

## 📁 Project Structure

fifa-tweet-sentiment-analyzer/
├── app.py # Main Gradio application
├── requirements.txt # Dependencies for Hugging Face
├── logistic_regression.pkl # Trained sentiment classifier
└── tfidf_vectorizer.pkl # TF-IDF transformer used for vectorization

bash
Copy
Edit

## 🛠 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/fifa-tweet-sentiment-analyzer.git
cd fifa-tweet-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
📚 NLTK Resources Required
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
👨‍💻 Author
Ansal CM
📝 Internship Project: Sentiment Analysis on FIFA Tweets
🔗 View on Hugging Face

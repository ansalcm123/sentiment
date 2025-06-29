# ⚽ FIFA Tweet Sentiment Analyzer

This project is a beginner-friendly **Machine Learning web app** that predicts the **sentiment of tweets related to the FIFA World Cup** — using NLP techniques like TF-IDF and Logistic Regression. It was built as part of an internship project and deployed using **Gradio** and **Hugging Face Spaces**.

---

## 🚀 Live Demo

👉 [🌐 Try the Web App on Hugging Face](https://huggingface.co/spaces/ansalcm/fifa-tweet-sentiment-analyzer)

---

## 📓 Project Notebook

👉 [📘 View Full Google Colab Notebook](https://colab.research.google.com/drive/1PygmPwNJTBMOMWylOPpUAR7VEbRFnIg3?usp=sharing)

This notebook includes:
- Data loading and cleaning
- Sentiment distribution visualization
- Text preprocessing
- Model training and evaluation
- Model saving for deployment

---

## 🔧 Technologies Used

- Python
- Scikit-learn
- NLTK
- Gradio
- Google Colab
- Hugging Face Spaces

---

## 🧠 How It Works

1. The user inputs a tweet related to the FIFA World Cup.
2. The app preprocesses the text (cleaning, tokenizing, lemmatizing).
3. It uses a TF-IDF vectorizer to transform the text into numerical form.
4. The Logistic Regression model predicts the sentiment: **Positive**, **Neutral**, or **Negative**.
5. The result is displayed with a confidence score in a clean Gradio interface.

---

## 📁 Project Structure

fifa-tweet-sentiment-analyzer/
├── app.py # Gradio web app
├── requirements.txt # Python dependencies
├── logistic_regression.pkl # Trained sentiment classifier
└── tfidf_vectorizer.pkl # TF-IDF vectorizer for input transformation

yaml
Copy
Edit

---

## 💻 Run This App Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/fifa-tweet-sentiment-analyzer.git
cd fifa-tweet-sentiment-analyzer

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the app
python app.py
📚 Required NLTK Downloads
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
👨‍💻 Author
Ansal CM
Internship Project: Sentiment Analysis using Machine Learning
🛰️ Hosted on Hugging Face Spaces


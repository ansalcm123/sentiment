import joblib
import gradio as gr
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("logistic_regression.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()  # replaced word_tokenize to avoid punkt_tab error

    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

# Prediction function
def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = max(probabilities)

    return f"🧠 Prediction: {prediction}\n🔍 Confidence: {confidence:.2%}"

# Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a FIFA tweet..."),
    outputs="text",
    title="⚽ FIFA Tweet Sentiment Analyzer",
    description="Enter a tweet related to the FIFA World Cup to predict its sentiment."
)

if __name__ == "__main__":
    interface.launch()

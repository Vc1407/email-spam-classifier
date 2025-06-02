import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load Model & Vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = word_tokenize(text)  # Tokenization
    text = [word for word in text if word.isalnum()]  # Remove punctuations
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(text)

# Streamlit UI
st.title("Spam Detector")
st.subheader("Enter a message to check if it's spam or not")

# Text Input
user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess & Transform Input
        processed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(transformed_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"

        # Display Result
        st.write(f"Prediction: **{result}**")


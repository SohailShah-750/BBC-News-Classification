
import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("bbc_news_model.pkl")
vectorizer = joblib.load("bbc_news_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 BBC News Classification")
st.write("AI-powered news category prediction")

# User input
news_text = st.text_area(
    "Paste your news article here:",
    height=250,
    placeholder="Enter news article..."
)

# Prediction
if st.button("Predict Category"):
    if news_text.strip():
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = max(probabilities) * 100

        st.success(f"Predicted Category: {prediction.title()}")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")
    else:
        st.warning("Please enter a news article first.")


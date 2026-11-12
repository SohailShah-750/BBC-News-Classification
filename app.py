
import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BBC News AI",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1726;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}
.stButton button {
    background-color: #00bfa6;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #009e8a;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
def load_files():
    model = joblib.load("bbc_news_model.pkl")
    vectorizer = joblib.load("bbc_news_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_files()


# ---------------- HEADER ----------------
st.title("📰 BBC News AI Classifier")
st.write("AI-powered news category prediction using Naive Bayes.")
st.divider()

# ---------------- DASHBOARD ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Algorithm", "Multinomial Naive Bayes")

with col2:
    st.metric("Vectorizer", "CountVectorizer")

st.divider()

# ---------------- NEWS INPUT ----------------
st.subheader("📄 Enter News Article")

news_text = st.text_area(
    "Paste your news article here:",
    height=250,
    placeholder="Enter or paste your news article..."
)

# ---------------- PREDICTION ----------------
if st.button("🚀 Analyze News", use_container_width=True):

    if news_text.strip():

        try:
            text_vector = vectorizer.transform([news_text])

            prediction = model.predict(text_vector)[0]

            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100

            st.success("News Analysis Completed!")

            st.subheader("🎯 Prediction Result")
            st.title(f"📌 {prediction}")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            # Probability chart
            categories = model.classes_

            probability_df = pd.DataFrame({
                "Category": categories,
                "Probability": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.subheader("📊 Category Probabilities")

            st.bar_chart(
                probability_df.set_index("Category")
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    else:
        st.warning("Please enter a news article first.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Python, Scikit-learn, Naive Bayes & Streamlit")

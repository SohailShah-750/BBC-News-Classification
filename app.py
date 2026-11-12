
import streamlit as st
import joblib
import pandas as pd

from datetime import datetime

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

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

# ---------------- SIDEBAR NAVIGATION ----------------

with st.sidebar:
    st.title("📰 BBC INTELLIGENCE")
    st.caption("AI-Powered News Platform")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "AI News Classifier",
            "Prediction History",
            "Analytics",
            "About"
        ]
    )

    st.divider()

    st.caption("POWERED BY")
    st.markdown("🤖 Multinomial Naive Bayes")
    st.markdown("⚡ Scikit-learn")
    st.markdown("🎨 Streamlit")

    st.divider()

    st.caption("Version 1.0.0")
    st.caption("© BBC News AI")



# ---------------- HEADER ----------------

# ---------------- OVERVIEW PAGE ----------------

if page == "Overview":

    st.title("📰 BBC News Intelligence")
    st.markdown(
        "### Welcome to your AI-powered News Intelligence Platform"
    )

    st.write(
        "Classify news articles into categories using Machine Learning."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Algorithm", "Naive Bayes")

    with col2:
        st.metric("Text Processing", "CountVectorizer")

    with col3:
        st.metric("Model Features", f"{model.n_features_in_}")

    st.divider()

    st.subheader("🚀 Platform Capabilities")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🧠 AI News Classification")
        st.write("Predict the category of any news article.")

        st.markdown("### 📊 Smart Analytics")
        st.write("Explore prediction confidence and category insights.")

    with c2:
        st.markdown("### 📚 Prediction History")
        st.write("Keep track of previous news predictions.")

        st.markdown("### 📥 Export Reports")
        st.write("Download prediction records for further analysis.")

    st.info("Select AI News Classifier from the sidebar to get started.")


# ---------------- AI NEWS CLASSIFIER PAGE ----------------

elif page == "AI News Classifier":

    st.title("🧠 AI News Classifier")
    st.markdown("Analyze any news article using our trained AI model.")

    st.divider()

    st.subheader("📝 News Article Input")

    news_text = st.text_area(
        "Paste your news article",
        height=250,
        placeholder="Paste the complete news article here..."
    )

    if st.button("🚀 Analyze News", use_container_width=True):

        if news_text.strip():

            try:
                text_vector = vectorizer.transform([news_text])

                prediction = model.predict(text_vector)[0]

                probabilities = model.predict_proba(text_vector)[0]
                confidence = max(probabilities) * 100

                st.success("News Analysis Completed!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Category", prediction)

                with col2:
                    st.metric("Confidence Score", f"{confidence:.2f}%")

                probability_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": probabilities * 100
                })

                probability_df = probability_df.sort_values(
                    by="Probability",
                    ascending=False
                )

                st.subheader("📊 Category Probability Analysis")

                st.bar_chart(
                    probability_df.set_index("Category")
                )

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        else:
            st.warning("Please enter a news article first.")


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

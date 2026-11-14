import streamlit as st
import joblib
import pandas as pd

from datetime import datetime

# ---------------- PAGE CONFIG (sabse pehle hona chahiye) ----------------
st.set_page_config(
    page_title="BBC News AI",
    page_icon="📰",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

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
@st.cache_resource
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
            "CSV Upload",
            "Prediction History",
            "Analytics",
            "About"
        ]
    )

# =====================================================
# OVERVIEW PAGE
# =====================================================
if page == "Overview":

    st.title("📰 BBC News Intelligence")
    st.write("Classify news articles into categories using Machine Learning.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Algorithm", "Naive Bayes")

    with col2:
        st.metric("Text Processing", "CountVectorizer")

    with col3:
        st.metric("Model Features", "5,000")

    st.divider()

    st.subheader("🚀 Platform Capabilities")

    c1, c2 = st.columns(2)

    with c1:
        st.info("📰 AI News Classification")
        st.info("📂 Bulk CSV News Classification")

    with c2:
        st.info("📊 Prediction History")
        st.info("📈 Analytics Dashboard")

# =====================================================
# AI NEWS CLASSIFIER PAGE
# =====================================================
elif page == "AI News Classifier":

    st.title("📰 AI News Classifier")
    st.write("Enter a news article to predict its category.")

    news_text = st.text_area(
        "Paste your news article here:",
        height=250,
        placeholder="Enter or paste your news article..."
    )

    if st.button("🚀 Analyze News"):

        if news_text.strip():

            try:
                # Vectorization
                text_vector = vectorizer.transform([news_text])

                # Prediction
                prediction = model.predict(text_vector)[0]

                # Probabilities
                probabilities = model.predict_proba(text_vector)[0]

                st.success(f"Predicted Category: {prediction}")

                probability_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": (probabilities * 100).round(2)
                })

                probability_df = probability_df.sort_values(
                    by="Probability",
                    ascending=False
                )

                st.subheader("📊 Category Probabilities")

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(probability_df.set_index("Category"))

                # Save to history
                st.session_state.prediction_history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "News Preview": news_text[:100] + ("..." if len(news_text) > 100 else ""),
                    "Predicted Category": prediction,
                    "Confidence (%)": round(float(probabilities.max() * 100), 2)
                })

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        else:
            st.warning("Please enter a news article first.")

# =====================================================
# CSV UPLOAD PAGE
# =====================================================
elif page == "CSV Upload":

    st.title("📂 Bulk News Classification")
    st.write("Upload a CSV file to classify multiple news articles.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            # Check CSV columns
            if "text" in df.columns:
                df["text"] = df["text"].fillna("").astype(str)

            elif "title" in df.columns and "content" in df.columns:
                df["text"] = (
                    df["title"].fillna("").astype(str)
                    + " "
                    + df["content"].fillna("").astype(str)
                )

            else:
                st.error(
                    "CSV must contain 'text' OR both 'title' and 'content' columns."
                )
                st.stop()

            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head(), use_container_width=True)

            st.metric("Total News Articles", len(df))

            if st.button("🚀 Classify All News", use_container_width=True):

                # Vectorization
                text_vector = vectorizer.transform(df["text"])

                # Predictions + Probabilities
                predictions = model.predict(text_vector)
                probabilities = model.predict_proba(text_vector)

                df["Predicted Category"] = predictions
                df["Confidence"] = (probabilities.max(axis=1) * 100).round(2)

                # Accuracy agar actual category column ho
                if "category" in df.columns:

                    actual = df["category"].astype(str).str.strip().str.lower()
                    predicted = pd.Series(predictions).astype(str).str.strip().str.lower()

                    accuracy = (actual.values == predicted.values).mean() * 100

                    st.metric("🎯 Model Accuracy", f"{accuracy:.2f}%")

                st.success("All news articles classified successfully!")

                st.subheader("📊 Classification Results")
                st.dataframe(df, use_container_width=True)

                # Category distribution chart
                st.subheader("📊 Category Distribution")
                st.bar_chart(df["Predicted Category"].value_counts())

                # Probability Table
                st.subheader("📈 Category Probabilities")
                probability_df = pd.DataFrame(
                    probabilities,
                    columns=model.classes_
                )
                st.dataframe(probability_df.round(3), use_container_width=True)

                # Download Results
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv,
                    file_name="bbc_classified_news.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"CSV Processing Error: {e}")

# =====================================================
# PREDICTION HISTORY PAGE
# =====================================================
elif page == "Prediction History":

    st.title("📚 Prediction History")
    st.write("Review your previous AI news predictions.")

    st.divider()

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        st.metric("Total Predictions", len(history_df))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download History CSV",
            data=csv,
            file_name="bbc_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info("No prediction history available yet. Analyze a news article first.")

# =====================================================
# ANALYTICS PAGE
# =====================================================
elif page == "Analytics":

    st.title("📈 Analytics Dashboard")
    st.write("Insights from your predictions in this session.")

    st.divider()

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Total Predictions", len(history_df))

        with c2:
            st.metric(
                "Average Confidence",
                f"{history_df['Confidence (%)'].mean():.2f}%"
            )

        st.subheader("📊 Predictions by Category")
        st.bar_chart(history_df["Predicted Category"].value_counts())

        st.subheader("📉 Confidence Over Time")
        st.line_chart(history_df["Confidence (%)"])

    else:
        st.info("No data yet. Analyze some news articles first.")

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":

    st.title("ℹ️ About")
    st.write(
        "This app classifies BBC news articles into categories "
        "using a Multinomial Naive Bayes model with CountVectorizer."
    )

    st.subheader("🛠️ Tech Stack")
    st.write("- Python\n- Scikit-learn\n- Streamlit\n- Pandas\n- Joblib")

    st.subheader("📂 Categories")
    st.write(", ".join(str(c) for c in model.classes_))

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Python, Scikit-learn & Streamlit")

         
# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Python, Scikit-learn, Naive Bayes & Streamlit")


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


# ---------------- HEADER ----------------

# ---------------- OVERVIEW PAGE ----------------

if page == "Overview":

    st.title("📰 BBC News Intelligence")
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

if st.button("🚀 Analyze News"):

    if news_text.strip():

        text_vector = vectorizer.transform([news_text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        st.success(f"Predicted Category: {prediction}")

        st.write("Category Probabilities:")

        probability_df = pd.DataFrame({
            "Category": model.classes_,
            "Probability": probabilities
        })

        st.dataframe(probability_df)

    else:
        st.warning("Please enter a news article first.")
               

                           # Calculate Accuracy

                if "category" in df.columns:

                    actual = (
                        df["category"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    predicted = (
                        pd.Series(predictions)
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    accuracy = (
                        actual == predicted
                    ).mean() * 100

                    st.metric(
                        "🎯 Model Accuracy",
                        f"{accuracy:.2f}%"
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

# -------------- PREDICTION HISTORY PAGE ----------------

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

    else:
        st.info("No prediction history available yet. Analyze a news article first.")




# ---------------- CSV UPLOAD PAGE ----------------


# ---------------- CSV UPLOAD PAGE ----------------

elif page == "CSV Upload":

    st.title("📂 Bulk News Classification")

    st.write(
        "Upload a CSV file to classify multiple news articles."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            # Read CSV File
            df = pd.read_csv(uploaded_file)

            # Check CSV columns
            if "text" in df.columns:

                df["text"] = (
                    df["text"]
                    .fillna("")
                    .astype(str)
                )

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

            # Uploaded Data
            st.subheader("📋 Uploaded Data")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            st.metric(
                "Total News Articles",
                len(df)
            )

            # Classify All News
            if st.button(
                "🚀 Classify All News",
                use_container_width=True
            ):

                # Vectorization
                text_vector = vectorizer.transform(
                    df["text"]
                )

                # Predictions
                predictions = model.predict(
                    text_vector
                )

                # Probabilities
                probabilities = model.predict_proba(
                    text_vector
                )

                # Add Predictions
                df["Predicted Category"] = predictions

                # Confidence Score
                df["Confidence"] = (
                    probabilities.max(axis=1) * 100
                ).round(2)

                # Accuracy if actual category exists
                if "category" in df.columns:

                    actual = (
                        df["category"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    predicted = (
                        pd.Series(predictions)
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    accuracy = (
                        actual == predicted
                    ).mean() * 100

                    st.metric(
                        "🎯 Model Accuracy",
                        f"{accuracy:.2f}%"
                    )

                # Success Message
                st.success(
                    "All news articles classified successfully!"
                )

                # Results
                st.subheader("📊 Classification Results")

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # Probability Table
                st.subheader("📈 Category Probabilities")

                probability_df = pd.DataFrame(
                    probabilities,
                    columns=model.classes_
                )

                st.dataframe(
                    probability_df.round(3),
                    use_container_width=True
                )

                # Download Results
                csv = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv,
                    file_name="bbc_classified_news.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"CSV Processing Error: {e}"
            )


# ---------------- PREDICTION HISTORY PAGE ----------------

elif page == "Prediction History":

         
# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Python, Scikit-learn, Naive Bayes & Streamlit")

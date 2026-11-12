
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="BBC News AI",
    page_icon="📰",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111c30;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #2dd4bf;
}
.subtitle {
    color: #94a3b8;
    font-size: 17px;
}
.result-card {
    background: #14243a;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #24516a;
}
</style>
""", unsafe_allow_html=True)

# Load saved AI files
@st.cache_resource
def load_files():
    model = joblib.load("bbc_news_model.pkl")
    vectorizer = joblib.load("bbc_news_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_files()

# Sidebar
with st.sidebar:
    st.title("📰 BBC AI")
    st.markdown("---")
    st.write("### Navigation")
    st.write("🏠 News Classifier")
    st.write("🤖 Multinomial Naive Bayes")
    st.markdown("---")
    st.caption("AI News Intelligence System")

# Main dashboard
st.markdown(
    '<div class="main-title">News Intelligence Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Classify news articles using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste your article below",
    height=220,
    placeholder="Paste news headline or article content here..."
)

if st.button("🚀 Analyze News", use_container_width=True):

    if news_text.strip():

        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]

        confidence = max(probabilities) * 100

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.markdown(
            f'<div class="result-card">'
            f'<h2 style="color:#2dd4bf;">{prediction.title()}</h2>'
            f'<p>Predicted News Category</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

        st.progress(float(confidence / 100))

        st.subheader("📊 Category Probabilities")

        chart_data = pd.DataFrame({
            "Category": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )

    else:
        st.warning("Please enter a news article.")

st.markdown("---")
st.caption("Powered by Multinomial Naive Bayes + CountVectorizer")

        st.success(f"Predicted Category: {prediction.title()}")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")
    else:
        st.warning("Please enter a news article first.")


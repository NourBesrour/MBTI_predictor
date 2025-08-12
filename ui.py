import streamlit as st
import joblib
import numpy as np

# ====== LOAD MODEL & VECTORIZER ======
model = joblib.load("multinomial_nb_model_ei_1_updated.pkl")
vectorizer = joblib.load("count_vectorizer_ei_1.pkl")

# ====== STREAMLIT APP ======
st.title("MBTI E/I Predictor 🌐")
st.write("Enter a text, and the model will predict if it is **Extrovert (E)** or **Introvert (I)**.")

# Text input
user_input = st.text_area("Enter your text here:", "")

if st.button("Predict"):
    if user_input.strip():
        # Vectorize input
        X_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]

        # Output mapping
        label_map = {0: "Introvert (I)", 1: "Extrovert (E)"}

        # Show result
        st.subheader("Prediction:")
        st.write(f"**{label_map[prediction]}**")
        st.write(f"Confidence: {proba[prediction]*100:.2f}%")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.caption("Model: MultinomialNB | Vectorizer: CountVectorizer | Updated with partial_fit")

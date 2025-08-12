import streamlit as st
import joblib
import numpy as np

# ===== LOAD MODEL & VECTORIZER =====
model = joblib.load("multinomial_nb_model_ei_1_updated.pkl")
vectorizer = joblib.load("count_vectorizer_ei_1.pkl")

# ===== STREAMLIT UI =====
st.title("MBTI E/I Predictor 🌐")
st.write("Enter text and see if the model predicts **Extrovert (E)** or **Introvert (I)**.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]

        label_map = {0: "Introvert (I)", 1: "Extrovert (E)"}

        st.subheader("Prediction:")
        st.write(f"**{label_map[prediction]}**")
        st.write(f"Confidence: {proba[prediction]*100:.2f}%")
    else:
        st.warning("Please enter some text.")

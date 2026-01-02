import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.layers import TextVectorization

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

def simple_cleaner(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('twitter_sentiment_model.keras')
    
    with open("vectorizer.pkl", "rb") as f:
        v_data = pickle.load(f)
    
    vectorizer = TextVectorization.from_config(v_data['config'])
    
    vectorizer.adapt(["dummy"]) 
    
    vectorizer.set_weights(v_data['weights'])
    
    return model, vectorizer


try:
    model, vectorize_layer = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading model or vectorizer: {e}")


st.title("🐦 Twitter Sentiment Analysis AI")
st.markdown("""
This application uses a **Bidirectional LSTM** model trained on **1.5 Million tweets** to predict the sentiment of your text with **83% accuracy**.
""")

st.subheader("Enter your tweet below:")
user_input = st.text_area("Tweet Input", placeholder="Type your message here...", height=150)

if st.button("Analyze Sentiment"):
    if not assets_loaded:
        st.error("Model not found! Please ensure 'twitter_sentiment_model.keras' and 'vectorizer.pkl' are in the same folder.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = simple_cleaner(user_input)
        
        vectorized_text = vectorize_layer([cleaned_text])
        
   
        prediction = model.predict(vectorized_text, verbose=0)[0][0]
        
        st.divider()
        st.subheader("Analysis Result:")
        
        if prediction >= 0.5:
            st.success(f"**Sentiment: Positive** 😊")
            st.info(f"Confidence Score: {prediction:.2f}")
        else:
            st.error(f"**Sentiment: Negative** ☹️")
            st.info(f"Confidence Score: {prediction:.2f}")


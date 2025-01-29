import streamlit as st
import xgboost as xgb
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random

os.chdir('C://Users//j_chr//OneDrive//Desktop//Projects//essay grader//')

df = pd.read_csv("ielts_writing_dataset.csv")

# Function to load the model
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgb_essay_grader.json')
    return model

random_prompt = df['Question'].sample(n=1).values[0]

# Function to preprocess the essay text input and extract features
# This should match whatever preprocessing was done during model training
def preprocess_essay(question_text, essay_text):
    X = question_text + " " + essay_text
    X = [X]

    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(X)
    return features

# Main Streamlit app
def main():
    st.title("Essay Grading System")

    st.write("IELTS Question")
    st.write(random_prompt)
    
    # Load the model
    model = load_model()
    
    # Input: Essay Text from the user
    essay_text = st.text_area("Enter your essay here:")

    if st.button("Grade Essay"):
        if essay_text:
            # Preprocess the essay
            features = preprocess_essay(random_prompt, essay_text)
            # Make the prediction (make sure to reshape it if needed)
            prediction = model.predict([features])
            st.write(f"Predicted grade: {prediction[0]:.2f}")
        else:
            st.error("Please enter an essay to grade.")

if __name__ == "__main__":
    main()

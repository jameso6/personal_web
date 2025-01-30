import streamlit as st
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

# Load your data


os.chdir('C://Users//j_chr//OneDrive//Desktop//Projects//essay grader//')

df = pd.read_csv("ielts_writing_dataset.csv")


# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
# Fit the vectorizer on the training data. Make sure to replace this with the actual training data.
# vectorizer.fit(train_data)  # You should fit on your actual training data here.

# Prepare the data
essays = df['Essay']
questions = df['Question']
y = df['Overall']

X = [q + " " + e for q, e in zip(questions, essays)]
random_prompt = df['Question'].sample(n=1).values[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Function to load the model
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgb_essay_grader.json')
    return model

# Function to preprocess the essay text input and extract features
def preprocess_essay(question_text, essay_text):
    X = question_text + " " + essay_text
    features = vectorizer.transform([X])  # Use transform instead of fit_transform
    return features



# Main Streamlit app
def main():
    st.title("Essay Grading System")

    # To retain the true state after clicking buttons
    # if 'clicked' not in st.session_state:
    #     st.session_state.clicked = False

    # def click_button():
    #     st.session_state.clicked = True

    if 'prompt' not in st.session_state:
        st.session_state.prompt = random_prompt

    def refresh_prompt():
        st.session_state.prompt = random_prompt

    # if st.button("IELTS Question", on_click=click_button):
        # st.write(random_prompt)
    st.write("IELTS Question:", st.session_state.prompt)
    
    # Load the model
    model = load_model()
        
    # Input: Essay Text from the user
    essay_text = st.text_area("Enter your essay here:")

    if st.button("Grade Essay"):
        if essay_text:
            # Preprocess the essay
            features = preprocess_essay(st.session_state.prompt, essay_text)
            # Convert sparse matrix to dense array if necessary
            prediction = model.predict(features.toarray())  # or just use features if the model accepts sparse
            st.write(f"Predicted grade: {prediction[0]:.2f}")
        else:
            st.error("Please enter an essay to grade.")
    
    if st.button('New Prompt'):
        refresh_prompt()
        st.rerun()


if __name__ == "__main__":
    main()
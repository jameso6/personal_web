import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

with open('style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.title("James Oblea")
st.subheader("Data Analyst")

# Provide a download button for the PDF
pdf_file_path = "JamesOblea_Resume_2025.pdf"

with open(pdf_file_path, "rb") as pdf_file:
    st.download_button(
        label="Download My Resume",
        data=pdf_file,
        file_name="JamesOblea_Resume_2025.pdf",
        mime="application/pdf"
    )

# Contact Information
st.header("Contact Information")
st.write("📍 Location: Calgary, AB")
st.write("📧 Email: jameschristian.oblea@gmail.com")
st.write("📱 Phone: (403) 919-8706")
st.write("🌐 LinkedIn: ca.linkedin.com/in/james-oblea")

# Summary Section
st.header("Highlights of Qualification")
st.write("• **Master’s graduate** with over 1+ year of experience in visualizing company KPIs and directing business decisions using data.")
st.write("• Experienced in building machine learning models to facilitate academic research – one of which was presented in the Alberta Children’s Hospital Research Institute research symposium.")
st.write("• **Languages & Frameworks**: Python, R, SQL (Postgres, MySQL, MSSQL), Java, JavaScript, HTML/CSS, Node.js, Streamlit, Flask")
st.write("• **Analysis Tools**: PowerBI, Tableau")
st.write("• **Libraries**: pandas, NumPy, Matplotlib, Plotly, Seaborn, Sci-kit learn, Tensor Flow, Keras, PyTorch")

# Experience Section
st.header("Employment Experience")
st.write("**Data Analyst Intern, - TOSSA Sustainability (Calgary, AB)** | May 2024 – August 2024")
st.write('''
         • Generated a historical database of employee wages and working hours across Alberta by extracting and processing over 2000
rows of data from 4 open government sources using Statistics Canada’s API, web scraping tools, and GitHub Actions.
         ''')
st.write('''
         • Employed PowerBI to create 13 dynamic visualizations illustrating the comparative employment wages and working
hours in different Alberta industry sectors and demographics to aid further analyses.
         ''')

st.write("**Business Analyst, - VantEdge Logistics Inc. (Calgary, AB)** | June 2022 – June 2023")
st.write('''
         • Established a new workflow system for business operations and enabled business success amidst declining resources.
         ''')
st.write('''
         • Liaison between operation and development teams of 20 people and managed the assignment of 30 weekly
Microsoft DevOps tasks and tickets ensuring customer success.
         ''')
st.write('''
         • Deployed a successful customer relationship management tool in HubSpot to track business development and
customer journey of over 20 clients.
         ''')

st.write("**Math Tutor, - MathPro Learning Centre (Calgary, AB)** | February 2019 – November 2023")
st.write('''
         • Educated and mentored over 50 students across different levels of school and significantly improved academic performance.
         ''')
st.write('''
         • Collaborated with management in establishing an online tutoring module to continue business operations during COVID.
         ''')


# Education Section
st.header("Education")
st.write("**Master of Data Science and Analytics** - University of Calgary (Calgary, AB) | 2024")
st.write("**Bachelor of Science in Mathematics and Statistics** - University of Calgary (Calgary, AB) | 2020")
st.write("**Bachelor of Science in Psychology** - University of Calgary (Calgary, AB) | 2020")


# # Skills Section
# st.header("Skills")
# skills = ["Python", "R", "SQL(Postgres, MySQL, MSSQL)", "Machine Learning", "Power BI", "Tableau","Project Management", "JavaScript", "HTML/CSS",]
# st.write(", ".join(skills))

# Projects Section
st.header("Projects")
st.write("**Automating IELTS Essay Grading**, Python| 2024")
st.write("• Extracted and vectorized 1400 IELTS essay responses and questions in preparation for model training.")
st.write("• Built a Natural Language Processing XGBoost Regression model to predict essay scores with a mean square error of 0.70.")
st.write("[https://github.com/jameso6/personal_web.git](#)")


# Uncomment below to download dataset from Kaggle
# api = KaggleApi()
# api.authenticate()

# api.dataset_download_files('mazlumi/ielts-writing-scored-essays-dataset', path = '.', unzip=True)

# Load dataset
df = pd.read_csv("ielts_writing_dataset.csv")

# Data prep
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
    model = xgb.Booster()
    model.load_model('xgb_essay_grader.json')
    return model

# Function to preprocess the essay text input and extract features
def preprocess_essay(question_text, essay_text):
    X = question_text + " " + essay_text
    features = vectorizer.transform([X])  # Use transform instead of fit_transform
    return features


# Main Streamlit app
def main():
    # st.title("Essay Grading System")
    st.write("Enter your response to the prompt in the text box below.")

    # Initialize random prompt state
    if 'prompt' not in st.session_state:
        st.session_state.prompt = random_prompt

    # Function to refresh prompt
    def refresh_prompt():
        st.session_state.prompt = random_prompt

    st.write("IELTS Question:", st.session_state.prompt)
    
    # Load the model
    model = load_model()
        
    # Input: Essay Text from the user
    essay_text = st.text_area("Enter your essay here:")

    if st.button("Grade Essay"):
        if essay_text:
            # Preprocess the essay
            features = preprocess_essay(st.session_state.prompt, essay_text)

            # Predict score using model
            prediction = model.predict(features.toarray()) # ensure datatype is consistent with model
            st.write(f"Predicted grade: {prediction[0]:.2f}")
        else:
            st.error("Please enter an essay to grade.")
    
    if st.button('New Prompt'):
        refresh_prompt()
        st.rerun()



if __name__ == "__main__":
    main()

st.write("**Simulating Late-Game Strategies in Basketball**, Python| 2024")
st.write("• Extracted NBA play-by-play data from over 7000 games using Basketball Reference’s API to fit probability distributions of random variables..")
st.write("• Performed a Discrete Event Monte Carlo Simulation of late-game basketball possessions to test three different late-game strategies and determine the best strategy to win basketball games..")
st.write("[https://github.com/jameso6/personal_web.git](#)")

# Footer Section
st.markdown("---")
st.write("Thank you for reviewing my resume. Feel free to contact me for more information or inquiries.")

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#NLTK
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")

# Page Setup
st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

#  Modern Dark UI
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Main container */
.block-container {
    padding: 2rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
}

/* Inputs */
textarea, input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

/* File uploader */
.stFileUploader {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
}

/* Metric box */
div[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

/* Headers */
h1, h2, h3 {
    color: #60a5fa;
}

/* Card effect */
.css-1d391kg {
    background-color: #1e293b !important;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

#Header
st.markdown("""
<h1 style='text-align: center;'> EduSpark Hirelytics </h1>
<hr style='border:1px solid #334155'>
""", unsafe_allow_html=True)

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    - Measures resume vs job match  
    - Finds important keywords  
    - Helps improve resume  
    """)
    
    st.header("How It Works")
    st.write("""
    1. Upload Resume  
    2. Paste Job Description  
    3. Click Analyze  
    4. Get Score  
    """)

# Helper Functions 
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reder = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reder.pages:
            text = text + page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    
    return round(score, 2), resume_processed, job_processed

# Main App 
def main():

    col1, col2 = st.columns([2,1])

    with col1:
        uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=['pdf'])
        job_description = st.text_area("📝 Paste the job description", height=200)

        analyze_btn = st.button("Analyze Match")

    with col2:
        st.info("💡 Tip: Add full job description for better accuracy.")

    if analyze_btn:

        if not uploaded_file:
            st.warning("Please upload your resume")
            return

        if not job_description:
            st.warning("Please paste the job description")
            return

        with st.spinner("Analyzing your resume...."):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text:
                st.error("could not extract text from pdf. please try another pdf")
                return 

            similarity_score, resume_processed, job_processed = calculate_similarity(
                resume_text, job_description
            )

            # Result 
            st.subheader("📊 Results")
            st.metric("Match Score", f"{similarity_score:.2f}%")

            # Chart 
            fig, ax = plt.subplots(figsize=(6,0.5))
            colors = ['#ff4b4b','#ffa726','#0f9d58']
            color_index = min(int(similarity_score//33),2)

            ax.barh([0],[similarity_score],color=colors[color_index])
            ax.set_xlim(0,100)
            ax.set_xlabel("Match percentage")
            ax.set_yticks([])
            ax.set_title("Resume Job Match")

            st.pyplot(fig)

            # Feedback 
            if similarity_score < 40:
                st.warning("Low Match, consider improving your resume.")
            elif similarity_score < 70:
                st.info("Good Match. Can be improved.")
            else:
                st.success("Excellent Match!")

if __name__ == "__main__":
    main()
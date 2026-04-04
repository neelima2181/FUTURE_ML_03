import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocessing import clean_resume_text
from skill_extraction import extract_skills
from similarity_model import calculate_similarity_score
from ranking import rank_candidates, identify_missing_skills

# Set page config
st.set_page_config(page_title="Resume Screening & Ranking System", layout="wide")

# Title and Description
st.title("📄 AI-Powered Resume Screening & Candidate Ranking System")
st.markdown("""
    Automatically rank candidates based on their similarity to a Job Description (JD).
    This system uses NLP to match techniques, identify missing skills, and visualize top talent.
""")

# --- Helper Functions ---
def load_data():
    dataset_path = 'dataset/Resume.csv'
    if not os.path.exists(dataset_path):
        # Create a mock dataset if not found for testing
        st.warning("⚠️ Resume.csv not found in 'dataset/' directory. Creating a mock dataset for demonstration.")
        mock_data = {
            'Category': ['Data Science', 'Data Science', 'Data Science', 'HR', 'HR', 'IT', 'IT'],
            'Resume_str': [
                "I am a Python developer with experience in Machine Learning and SQL. Skilled in Pandas and Spark.",
                "Expert in AI and Data Science. I use Python, TensorFlow, and AWS.",
                "Data Analyst with SQL and Tableau skills. No Machine Learning experience.",
                "HR Manager with 5 years experience in Talent Acquisition and Recruitment.",
                "Human Resources Specialist focused on employee relations and payroll.",
                "IT Specialist with expertise in Cloud and Cybersecurity. SQL and Java developer.",
                "Full-stack developer with React, Node.js, and Java experience."
            ]
        }
        df = pd.DataFrame(mock_data)
        os.makedirs('dataset', exist_ok=True)
        df.to_csv(dataset_path, index=False)
        return df
    return pd.read_csv(dataset_path)

# --- Sidebar Inputs ---
st.sidebar.header("Job Search Criteria")
df = load_data()

# Select Category
categories = df['Category'].unique()
selected_category = st.sidebar.selectbox("Select Job Category", categories)

# Input Job Description
st.sidebar.subheader("Input Job Description")

sample_jds = {
    "Data Science Role": "Looking for a Python developer with expertise in Machine Learning, SQL, and Cloud. Must be familiar with Data Science techniques.",
    "HR Manager Role": "Seeking an experienced HR Manager with strong skills in Talent Acquisition, Employee Relations, and Payroll management.",
    "IT Specialist Role": "Hiring an IT Specialist with a background in Cybersecurity, Cloud infrastructure (AWS/Azure), and Java/SQL development.",
    "Custom (Type your own)": ""
}

selected_sample = st.sidebar.selectbox("Load Sample Job Description", list(sample_jds.keys()))

default_text = sample_jds[selected_sample]

job_description = st.sidebar.text_area("Required Skills/Job Role description", 
    value=default_text,
    height=200
)

# Process Button
if st.sidebar.button("Process Resumes"):
    if not job_description.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Processing Resumes..."):
            # 1. Filter candidates by category
            filtered_df = df[df['Category'] == selected_category].copy()
            
            if filtered_df.empty:
                st.error("No resumes found in this category.")
            else:
                # 2. Extract required skills from JD
                required_skills = extract_skills(job_description)
                
                # 3. Clean texts for similarity calculation
                cleaned_jd = clean_resume_text(job_description)
                
                # Process each filtered resume
                resumes_text_list = filtered_df['Resume_str'].tolist()
                cleaned_resumes_list = [clean_resume_text(r) for r in resumes_text_list]
                
                # 4. Similarity Scoring
                scores = calculate_similarity_score(cleaned_resumes_list, cleaned_jd)
                
                # 5. Extract Skills for each candidate and find missing skills
                candidate_skills_list = []
                missing_skills_list = []
                
                for r in resumes_text_list:
                    c_skills = extract_skills(r)
                    candidate_skills_list.append(", ".join(c_skills))
                    
                    missing = identify_missing_skills(c_skills, required_skills)
                    missing_skills_list.append(", ".join(missing))
                
                filtered_df['Extracted Skills'] = candidate_skills_list
                filtered_df['Missing Skills'] = missing_skills_list
                
                # 6. Rank Candidates
                ranked_candidates = rank_candidates(filtered_df, scores)
                
                # --- Success Display ---
                st.success(f"Processing complete! Found {len(ranked_candidates)} candidates in {selected_category}.")
                
                # --- High Score Highlight ---
                best_match = ranked_candidates.iloc[0]
                st.info(f"🏆 **Best Matching Candidate Found!** (Similarity: {best_match['Similarity Score']*100:.2f}%)")
                
                # --- Visualizations ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Candidate Rankings (Similarity Score)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Take top 10 for visualization
                    top_n = ranked_candidates.head(10)
                    # For names, use Index since we don't have a name column in this dataset (usually only ID)
                    sns.barplot(x=top_n['Similarity Score'], y=top_n.index.astype(str), palette='viridis', ax=ax)
                    ax.set_xlabel("Similarity Score")
                    ax.set_ylabel("Candidate ID (Row Index)")
                    ax.set_title(f"Top Matching Candidates in {selected_category}")
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Required Skills identified in JD")
                    st.write(required_skills if required_skills else "No specific skills identified.")
                
                # --- Results Table ---
                st.subheader("Full Ranking Details")
                
                # Format scores for display
                display_df = ranked_candidates[['Similarity Score', 'Extracted Skills', 'Missing Skills']].copy()
                display_df['Similarity Score'] = (display_df['Similarity Score'] * 100).map('{:.2f}%'.format)
                
                st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 Enter criteria and click 'Process Resumes' to see the results.")

# --- Footer ---
st.markdown("---")

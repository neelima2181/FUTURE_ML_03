# Resume Screening and Candidate Ranking System using NLP

## 📌 Project Overview
This project is an AI-powered recruitment tool designed to automate the initial phase of candidate screening. By leveraging Natural Language Processing (NLP), the system ranks resumes based on their similarity to a specific job description and identifies "Skill Gaps" to help recruiters make data-driven decisions.

## 🎯 Objective
- Automatically filter and rank candidates from large datasets.
- Calculate similarity scores between resumes and job requirements.
- Visualize candidate matching results for quick analysis.
- Identify missing skills in each candidate's profile compared to the job description.

## ⚙️ Workflow Explanation
1. **Data Loading**: The system reads the Kaggle Resume Dataset (CSV).
2. **Text Cleaning**: Resumes and JD are preprocessed (lowercase, noise removal, lemmatization).
3. **Skill Extraction**: `spaCy`'s NLP pipeline extracts technical & soft skills using keyword patterns.
4. **Vectorization**: `TF-IDF Vectorizer` converts text into numerical representations.
5. **Similarity Matching**: `Cosine Similarity` calculates the distance between the JD vector and candidate vectors.
6. **Ranking & Analysis**: Candidates are sorted by score, and missing skills are identified.
7. **Visualization**: Matplotlib/Seaborn charts display the top matches in the Streamlit UI.

## 🛠️ Technologies Used
- **Python**: Core logic and data processing.
- **Streamlit**: Web dashboard for interactive use.
- **spaCy**: NLP pipeline for entity/keyword extraction.
- **NLTK**: Text preprocessing and lemmatization.
- **Scikit-learn**: TF-IDF Vectorization and Cosine Similarity.
- **Matplotlib/Seaborn**: Data visualization.
- **Pandas/Numpy**: Data manipulation.

## 📊 Dataset Information
- **Source**: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- **Columns**:
  - `ID`: Unique identifier.
  - `Resume_str`: Raw resume text.
  - `Category`: Job role/category (e.g., HR, Data Science).

## 🚀 Getting Started

Follow these steps to set up and run the dashboard on your local machine:

1. **Install Python Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLP Models & Resources**:
   The system requires specific spaCy and NLTK resources:
   ```bash
   # Download spaCy model
   python -m spacy download en_core_web_sm

   # Download NLTK resources
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet');"
   ```

3. **Run the Dashboard**:
   Start the Streamlit application using Python (this avoids common Windows path issues):
   ```bash
   python -m streamlit run app.py
   ```
   *The app will automatically open in your default browser at `http://localhost:8501`.*

## 📸 Output Details
- **Ranked Candidate List**: Shows the most relevant candidates first.
- **Similarity Percentage**: A measure of how well the resume matches the JD.
- **Missing Skills**: List of required skills from the JD that weren't found in the resume.
- **Top Match Highlight**: Visual callout for the best-fit candidate.

## 🔮 Future Improvements
- **PDF/Docx Support**: Add direct file upload for individual resumes.
- **OCR Integration**: Handle image-based resumes using Tesseract.
- **Advanced NER**: Train custom spaCy models for specialized industry skills.
- **Deep Learning**: Use BERT or Transformer models for more semantic matching.

---
**Built for Internship & Portfolio Submission**

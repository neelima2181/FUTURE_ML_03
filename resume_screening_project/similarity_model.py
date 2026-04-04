from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_similarity_score(resumes_text_list, job_description_text):
    """
    Calculate similarity score between multiple resumes and a single job description.
    """
    # 1. Combine job description and resumes for vectorization
    tfidf_vectorizer = TfidfVectorizer()
    
    all_texts = [job_description_text] + resumes_text_list
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # 2. Extract job description vector and resumes vectors
    job_desc_vector = tfidf_matrix[0]
    resumes_vectors = tfidf_matrix[1:]
    
    # 3. Calculate cosine similarity
    similarity_scores = cosine_similarity(job_desc_vector, resumes_vectors)
    
    return similarity_scores.flatten()

if __name__ == "__main__":
    job_desc = "Looking for a Python developer with Machine Learning and SQL expertise."
    resumes = [
        "I am a Python developer with 5 years experience in ML and SQL.",
        "Experienced Java developer who knows SQL but not Python or ML."
    ]
    
    scores = calculate_similarity_score(resumes, job_desc)
    print(f"Similarity Scores: {scores}")

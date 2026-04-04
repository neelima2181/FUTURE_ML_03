import spacy
from spacy.matcher import PhraseMatcher

# List of skills (Technical and Soft Skills)
SKILL_DB = [
    'Python', 'Java', 'Machine Learning', 'SQL', 'HTML', 'CSS', 'JavaScript', 
    'Data Science', 'AI', 'Cloud', 'Communication', 'Leadership',
    'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'NLTK', 'spaCy',
    'Django', 'Flask', 'FastAPI', 'React', 'Angular', 'Vue', 'Node.js', 
    'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Docker', 'Kubernetes',
    'AWS', 'Azure', 'GCP', 'DevOps', 'Git', 'GitHub', 'CI/CD',
    'Hadoop', 'Spark', 'Tableau', 'PowerBI', 'Excel', 'Statistics',
    'Deep Learning', 'Computer Vision', 'NLP', 'Big Data', 'Software Development'
]

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_skills(text):
    """
    Extract skills from the provided text using PhraseMatcher.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Create patterns for matching
    patterns = [nlp.make_doc(skill) for skill in SKILL_DB]
    matcher.add("SKILLS", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        skills.add(span.text)
        
    return list(skills)

if __name__ == "__main__":
    test_resume = "I am a Python developer with experience in Machine Learning, SQL, and AWS. I also use Pandas, Numpy, and Matplotlib."
    print(f"Resume text: {test_resume}")
    print(f"Extracted Skills: {extract_skills(test_resume)}")

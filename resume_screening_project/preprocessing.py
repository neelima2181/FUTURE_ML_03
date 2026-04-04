import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
def download_nltk_data():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)

download_nltk_data()

def clean_resume_text(text):
    """
    Clean and preprocess resume text:
    - Lowercase
    - Remove URLs, emails, special characters
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    # 1. Convert to lowercase
    text = str(text).lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    
    # 4. Remove special characters and numbers (keeping some punctuation like '.' for sentences if needed, 
    # but for matching we usually strip most)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # 5. Tokenization
    tokens = word_tokenize(text)
    
    # 6. Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # Join back to string
    return " ".join(tokens)

if __name__ == "__main__":
    test_text = "Check out my profile at http://linkedin.com/in/user or email me at user@example.com. I love Python and Machine Learning!"
    print(f"Original: {test_text}")
    print(f"Cleaned:  {clean_resume_text(test_text)}")

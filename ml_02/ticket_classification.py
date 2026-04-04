import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def train_and_save():
    print("🚀 Starting ML Pipeline Training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('customer_support_tickets.csv')
    except FileNotFoundError:
        print("❌ Error: 'customer_support_tickets.csv' not found.")
        return

    # 2. Heuristic Priority Labeling (as per Task_2_Explanation.md)
    def assign_priority(text):
        critical_words = ['urgent', 'critical', 'down', 'crash', 'failure', 'broken', 'immediate']
        info_words = ['info', 'request', 'how to', 'question', 'help', 'query']
        
        text_lower = str(text).lower()
        if any(word in text_lower for word in critical_words):
            return 'High'
        elif any(word in text_lower for word in info_words):
            return 'Low'
        else:
            return 'Medium'

    # Ensure necessary columns exist. Assuming 'Ticket Description' and 'Ticket Type' (Category)
    # Adjust column names based on the common structure.
    # Searching for text columns...
    text_col = 'Ticket Description' if 'Ticket Description' in df.columns else df.columns[1]
    cat_col = 'Ticket Type' if 'Ticket Type' in df.columns else df.columns[2]

    print(f"📦 Using columns: Text='{text_col}', Category='{cat_col}'")

    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['Priority'] = df[text_col].apply(assign_priority)

    # 3. Model for Category
    X = df['cleaned_text']
    y_cat = df[cat_col]
    
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print("⚙️ Training Category Model...")
    cat_pipeline.fit(X_train, y_train_cat)
    cat_acc = accuracy_score(y_test_cat, cat_pipeline.predict(X_test))
    print(f"✅ Category Accuracy: {cat_acc:.2f}")

    # 4. Model for Priority
    y_pri = df['Priority']
    X_train_p, X_test_p, y_train_pri, y_test_pri = train_test_split(X, y_pri, test_size=0.2, random_state=42)

    pri_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print("⚙️ Training Priority Model...")
    pri_pipeline.fit(X_train_p, y_train_pri)
    pri_acc = accuracy_score(y_test_pri, pri_pipeline.predict(X_test_p))
    print(f"✅ Priority Accuracy: {pri_acc:.2f}")

    # 5. Save Artifacts
    joblib.dump(cat_pipeline, 'category_model.pkl')
    joblib.dump(pri_pipeline, 'priority_model.pkl')
    print("💾 Models saved to root folder.")

    # 6. Save Historical Metrics for Dashboard
    metrics = pd.DataFrame({
        'Actual_Category': y_test_cat,
        'Predicted_Category': cat_pipeline.predict(X_test),
        'Actual_Priority': y_test_pri,
        'Predicted_Priority': pri_pipeline.predict(X_test_p)
    })
    metrics.to_csv('historical_metrics.csv', index=False)
    print("📊 Historical metrics updated for dashboard.")

if __name__ == "__main__":
    train_and_save()

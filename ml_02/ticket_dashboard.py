import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# -------------------------------------------------------------
# 1. SETUP & PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Support Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection of CSS for premium UI vibes
st.markdown("""
    <style>
    /* Gradient Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1f005c, #5b0060, #870160, #ac255e, #ca485c, #e16b5c, #f39060, #ffb56b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 15px 0 rgba(225, 107, 92, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(225, 107, 92, 0.6);
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. LOAD MODELS & NLP GLOBALS
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        cat_model = joblib.load('category_model.pkl')
        pri_model = joblib.load('priority_model.pkl')
        return cat_model, pri_model
    except FileNotFoundError:
        return None, None

cat_model, pri_model = load_models()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_input_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# -------------------------------------------------------------
# 3. HEADER
# -------------------------------------------------------------
st.title("🎫 Smart Support Ticket Routing Engine")
st.markdown("Automate categorize and prioritize inbound customer signals to eliminate backlog.")
st.divider()

if not cat_model or not pri_model:
    st.error("⚠️ **Models not found.** Please run `python ticket_classification.py` first to train and export the latest models.")
    st.stop()

# -------------------------------------------------------------
# 4. MAIN LAYOUT
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["⚡ Single Ticket Prediction", "📉 Model Analytics", "📁 Bulk Upload & Analyze"])

with tab1:
    st.subheader("Simulate Incoming Ticket")
    ticket_text = st.text_area(
        "Paste the body of a customer email/complaint here:",
        height=200,
        placeholder="e.g. 'Hello, the primary database server is down and we are losing millions of dollars. Fix it urgently!'"
    )
    predict_click = st.button("Predict ➜ Routing & Priority")

    if predict_click:
        if not ticket_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("AI Processing Natural Language..."):
                # 1. Clean Text
                cleaned = clean_input_text(ticket_text)
                
                # 2. Predict
                predicted_cat = cat_model.predict([cleaned])[0]
                predicted_pri = pri_model.predict([cleaned])[0]
                
                st.markdown("---")
                st.subheader("🤖 AI Decision Output")
                
                # Visual Badges for Priority
                if predicted_pri == "High":
                    pri_color = "red"
                    pri_icon = "🚨"
                elif predicted_pri == "Medium":
                    pri_color = "orange"
                    pri_icon = "⚠️"
                else:
                    pri_color = "green"
                    pri_icon = "✅"
                    
                out_col1, out_col2 = st.columns(2)
                with out_col1:
                    st.markdown(f"**Target Department:**")
                    st.success(f"**{predicted_cat}** team")
                with out_col2:
                    st.markdown(f"**Priority Level:**")
                    st.markdown(f"<h3 style='color: {pri_color}; margin-top: -10px;'>{pri_icon} {predicted_pri.upper()}</h3>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📊 AI Confidence Metrics")
                # Calculate Probability arrays
                cat_classes = cat_model.classes_
                probas_cat = cat_model.predict_proba([cleaned])[0]
                prob_df = pd.DataFrame({"Probability": probas_cat}, index=cat_classes)
                prob_df = prob_df.sort_values(by="Probability", ascending=False).head(4)
                
                # Use columns to constrain the width from stretching completely across monitor
                graph_col, _ = st.columns([0.5, 0.5])
                with graph_col:
                    st.bar_chart(prob_df, height=300)
                
                st.markdown(f"**Cleaned Tokens used by AI:**")
                st.caption(cleaned)
                
                if predicted_pri == "High":
                    st.balloons()

with tab2:
    st.subheader("Historical Model Analytics")
    st.markdown("These dynamic graphs represent the latest benchmark training accuracies from our internal datasets.")
    
    try:
        metrics_df = pd.read_csv('historical_metrics.csv')
        
        st.markdown("### Category Classifier Alignment")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Actual Subject Distribution")
            st.bar_chart(metrics_df['Actual_Category'].value_counts(), color="#1e88e5")
            
        with colB:
            st.markdown("#### System Predicted Distribution")
            st.bar_chart(metrics_df['Predicted_Category'].value_counts(), color="#e53935")
            
        st.markdown("---")
        
        st.markdown("### Priority Routing Alignment")
        colC, colD = st.columns(2)
        with colC:
            st.markdown("#### True Routing Priorities")
            st.bar_chart(metrics_df['Actual_Priority'].value_counts(), color="#43a047")
            
        with colD:
            st.markdown("#### Model Assigned Priorities")
            st.bar_chart(metrics_df['Predicted_Priority'].value_counts(), color="#fbc02d")
            
    except Exception as e:
        st.error(f"Could not construct metrics graphs: {e}")

with tab3:
    st.subheader("Bulk File Processing & Analytics")
    st.markdown("Upload a CSV file containing multiple tickets to predict their Category and Priority instantly.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df_upload.head(3))
            
            # Ask user to select the text column
            text_column = st.selectbox("Which column contains the ticket text?", df_upload.columns)
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing tickets..."):
                    df_upload['cleaned_text'] = df_upload[text_column].apply(clean_input_text)
                    
                    df_upload['Predicted_Category'] = cat_model.predict(df_upload['cleaned_text'])
                    df_upload['Predicted_Priority'] = pri_model.predict(df_upload['cleaned_text'])
                    
                    # Drop the cleaned token text for the final output
                    df_upload = df_upload.drop(columns=['cleaned_text'])
                    
                    st.success(f"Successfully processed {len(df_upload)} tickets! See the dynamic metrics below.")
                    st.dataframe(df_upload)
                    
                    st.markdown("---")
                    
                    # Wrap the graphs in padded columns so they don't stretch indefinitely wide
                    padLeft, colA, colB, padRight = st.columns([0.1, 0.4, 0.4, 0.1])
                    with colA:
                        st.markdown("#### Predicted Category Spread")
                        st.bar_chart(df_upload['Predicted_Category'].value_counts(), color="#6b9eff", height=300)
                    with colB:
                        st.markdown("#### Predicted Priority Spread")
                        st.bar_chart(df_upload['Predicted_Priority'].value_counts(), color="#ff6b6b", height=300)
                    
                    # Provide download link
                    csv_export = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Labeled CSV",
                        data=csv_export,
                        file_name="predicted_tickets.csv",
                        mime="text/csv",
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")

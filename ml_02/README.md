# 🎫 Smart Support Ticket Classification Engine (NLP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Front--End-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

Welcome to the **Support Ticket Classification Engine**—a robust, production-ready system that leverages Natural Language Processing (NLP) to automatically categorize customer support requests and predict their urgency (Priority). 

This project aims to bridge the gap between high-volume customer signals and efficient response teams by eliminating manual backlog triage.

---

## 🚀 Key Features

- **🧠 Advanced NLP Pipeline**: Uses TF-IDF vectorization and Logistic Regression to understand ticket context.
- **🚨 Priority Heuristics**: Automatically flags critical operational failures (High Priority) from informational requests (Low Priority).
- **📉 Live Analytics Dashboard**: A premium Streamlit interface to visualize model confidence and performance distributions.
- **📁 Bulk CSV Processing**: Upload entire corporate datasets to generate instant ML predictions for your ticket backlogs.
- **📈 Exportable Results**: Download labeled datasets directly from the dashboard for downstream routing.

---

## 🛠 Project Structure

```text
├── ticket_classification.py  # ⚙️ MAIN: Train and Export ML Models
├── ticket_dashboard.py       # 🎨 FRONT-END: Streamlit Analytics Dashboard
├── customer_support_tickets.csv # 📊 DATA: Primary training dataset
├── category_model.pkl        # 💾 MODEL: Saved Category Classifier
├── priority_model.pkl        # 💾 MODEL: Saved Priority Classifier
├── historical_metrics.csv    # 📈 METRICS: Training performance benchmarks
└── requirements.txt          # 📦 DEPENDENCIES: Required Python libraries
```

---

## 🏗 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models (Initial Run)
To generate the latest model artifacts and predictive metrics, run the main classification pipeline:
```bash
python ticket_classification.py
```

### 3. Launch the AI Dashboard
Open the interactive prediction and analytics engine:
```bash
python -m streamlit run ticket_dashboard.py
```

---

## 📤 Step-by-Step GitHub Push Guide

Since your repository `FUTURE_ML_02` was created with a README on GitHub, follow these exact steps to push your project locally and **overwrite** the remote with your fresh code:

1. **Stage all changes**:
   ```bash
   git add .
   ```
2. **Commit your work**:
   ```bash
   git commit -m "Added ML Task 2 Support Ticket Classification Project"
   ```
3. **Set the correct remote** (already configured):
   ```bash
   git remote set-url origin https://github.com/neelima2181/FUTURE_ML_02.git
   ```
4. **FORCE PUSH to reconcile history**:
   ```bash
   git push -u origin main --force
   ```

---

## 📝 Performance Insights
The model currently achieves significant accuracy on the **Zenodo IT Support Dataset**, prioritizing critical server-down events and routing general billing queries to specialized financial support teams.

Developed with ❤️ as part of the Future ML Task 2.

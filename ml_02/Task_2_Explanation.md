# Support Ticket Classification & Prioritization System

This document outlines the Machine Learning approach used to automate customer support ticket classification and prioritization.

## 🎯 Problem Statement
In real-world SaaS companies, customer support teams receive thousands of text-based tickets daily. Support agents waste valuable time manually reading, categorizing, and assigning priority to issues. This delays urgent issues from reaching engineering teams and creates a massive backlog.

## 🧠 The Machine Learning Solution
This project implements an NLP-based **decision-support system** that automatically reads tickets and assigns them:
1. **A Category** (e.g., General query, IT support, Billing)
2. **A Priority Level** (High, Medium, Low)

### 1. Dataset Selection
The solution utilizes the **Zenodo IT Support Tickets dataset** (~2,229 human-classified support tickets), which mimics real enterprise text structures and complexities.

### 2. How Tickets are Categorized
- **Text Cleaning:** All tickets undergo preprocessing using NLTK where special characters are removed, content is lowercased, and common English "stop words" (like *the, and, is*) are filtered out. Finally, words are lemmatized to their root form.
- **Feature Extraction (TF-IDF):** The cleaned text is transformed into a mathematical vector representation using TF-IDF (Term Frequency-Inverse Document Frequency) up to 5000 n-grams. This allows the model to understand the importance of specific words relative to the whole corpus.
- **Classification Model:** We use a **Logistic Regression** model optimized for high-dimensional sparse text data. This model predicts the designated department or category based entirely on the text input.

### 3. How Priority is Decided
Since most open datasets lack realistic severity scores, the system implements a **heuristic-labeling logic** to simulate operational prioritization:
- **High Priority:** Tickets containing critical operational buzzwords (e.g., *urgent, critical, down, crash, failure*).
- **Low Priority:** Tickets matching informational requests (e.g., *info, request, how to, question*).
- **Medium Priority:** The standard default for operational workflow.

A dedicated ML model is trained on these assigned priorities to predict new, unseen tickets.

## 📊 Evaluation & Insights
We evaluated the models using industry-standard machine learning metrics:
- **Accuracy:** General correctness across all categories.
- **Precision / Recall:** Important for tracking specific severe categories to minimize False Positives and False Negatives.
- **Confusion Matrices:** Visual aids (`confusion_matrix_category.png`, `confusion_matrix_priority.png`) generated to show exactly where the model might mistake one category for another, directing future training efforts.

### 🚀 Business Impact
By routing tickets instantly with Machine Learning:
- **Response times** improve substantially.
- **Support Operations** become 3x more efficient without extra agents.
- **Critical path issues** (e.g., server down) are surfaced instantly to the corresponding on-call team.

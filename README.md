# 📩 Ultra Modern Spam Classifier  

[![Live App](https://img.shields.io/badge/%F0%9F%9A%80%20Live%20App-Streamlit-brightgreen?style=for-the-badge&logo=streamlit&logoColor=white)](https://spam-classifier-ml-app.streamlit.app/) 
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) 
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) 
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) 
[![NLTK](https://img.shields.io/badge/NLTK-purple?style=for-the-badge&logo=nltk&logoColor=white)](https://www.nltk.org/) 
[![TF-IDF](https://img.shields.io/badge/TF--IDF-blueviolet?style=for-the-badge)]() 
[![Matplotlib](https://img.shields.io/badge/Matplotlib-FF6F61?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/) 
[![Seaborn](https://img.shields.io/badge/Seaborn-77ACF1?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/) 
 


🔗 **Live Demo App** → [Click Here](https://spam-classifier-ml-app.streamlit.app/) 
---

## 📌 Problem Statement  

Spam messages waste time, spread phishing attacks, and steal personal data.  
The goal was to build an **AI-powered classifier** that can detect spam with **extremely high precision**, ensuring **no false positives** (safe messages wrongly flagged).  

---

## 🌟 Overview  

The **Ultra Modern Spam Classifier** is a **complete end-to-end NLP + Machine Learning project**, designed to classify SMS/Email messages into **Spam** or **Not Spam**.  

This project covers **everything from raw data to deployment**:  
✔️ Data collection  
✔️ Cleaning & preprocessing  
✔️ NLP feature engineering (**TF-IDF**)  
✔️ Training & tuning multiple ML models  
✔️ Performance evaluation  
✔️ Deployment on **Streamlit Cloud** with a **modern UI**  

🚀 With **98% accuracy** and **1.00 (100%) precision**, this app ensures that *zero spam slips through*.  
---

## 🧠 End-to-End Workflow  

### 1️⃣ Data Collection  
- Used a **Spam-Ham dataset** of thousands of SMS messages.  

### 2️⃣ Data Preprocessing  
- Lowercasing  
- Removing punctuation, numbers, special characters  
- Tokenization  
- Stopword removal (using NLTK)  
- Stemming (Porter Stemmer)  

### 3️⃣ Feature Engineering  
- Applied **TF-IDF Vectorization** (Term Frequency - Inverse Document Frequency)  
- Converts text into a sparse numerical representation  

### 4️⃣ Model Training & Comparison  
->> 🔥 Machine Learning Algorithms Explored  

During experimentation, multiple algorithms were tested before finalizing the best-performing model.  
- ✅ **Multinomial Naive Bayes** (Chosen Model)  
- 🌲 **Random Forest Classifier**  
- 📈 **Logistic Regression**  
- 📊 **Support Vector Machine (SVM)**  
- 🔍 **K-Nearest Neighbors (KNN)**  
- 🧠 **Decision Tree Classifier**  
- ⚡ **XGBoost Classifier**  
- 🚀 **LightGBM Classifier**  
- 🔮 **Gradient Boosting Classifier**  
- 🎯 **AdaBoost Classifier**  
- 🔗 **Extra Trees Classifier**  
- 🌀 **Stochastic Gradient Descent (SGD) Classifier**  
- 🤖 **Perceptron (Single-layer Neural Network)**  
- 🧬 **MLP Classifier (Multi-layer Perceptron / Deep Neural Network)**  

> After benchmarking, **Multinomial Naive Bayes** emerged as the most efficient algorithm for spam classification with high accuracy and fast training speed.

6️⃣ Model Improvement & Optimization

After selecting the model algorithm, several steps were implemented to enhance performance and improve reliability:

<div style="margin-left: 20px;">
🔹 1. Max-Min Scaling / Standardization

Rescale features to a consistent range or normalize them for better convergence.

Impact: Improves accuracy but may slightly reduce precision.

🔹 2. TF-IDF Parameter Tuning

Adjust max_features, min_df, and max_df to select informative words.

Experiment with n-grams (unigrams, bigrams, trigrams) to capture context.

🔹 3. Feature Engineering Enhancements

Create new features:

Message length

Special character count

Numeric word frequency

Include positional or capitalization ratio features for additional signal.

🔹 4. Ensemble & Hybrid Methods

Combine models using:

Voting Classifier

Bagging

Boosting

Mix Multinomial Naive Bayes with Random Forest or XGBoost for improved results.

🔹 5. Hyperparameter Optimization

Fine-tune parameters using:

Grid Search

Randomized Search

Bayesian Optimization

Optimize multiple metrics:

Accuracy, Precision, Recall, F1-score

🔹 6. Cross-Validation

Implement k-fold cross-validation to validate the model on multiple splits.

Ensures robustness and reduces overfitting.

🔹 7. Regularization Techniques

Apply L1 / L2 regularization for Logistic Regression or SVM to prevent overfitting.

🔹 8. Model Calibration

Calibrate predicted probabilities for better threshold selection and more reliable predictions.

</div>


### 5️⃣ Hyperparameter Tuning  
- GridSearchCV & Cross-validation used for optimal parameters  

### 6️⃣ Model Selection  
**Multinomial Naive Bayes** chosen because:  
- Lightweight & fast  
- Handles **TF-IDF sparse data** very well  
- Achieved **perfect precision (1.00)**  
- Random Forest was close but heavier for deployment  

### 7️⃣ Evaluation  

| Metric       | Score       |  
|--------------|-------------|  
| Accuracy     | ~98%        |  
| Precision    | **1.00 (100%)** |  
| Recall       | ~0.97       |  
| F1-Score     | ~0.98       |  

✅ Precision = 1.00 → **all predicted spam were truly spam**.  

### 8️⃣ Deployment  
- Built web app using **Streamlit**  
- Custom **CSS animations** (glassmorphism, gradients)  
- Deployed on **Streamlit Community Cloud**  

---

## 🎨 App Showcase  


### 🚨 Prediction Example  
- ✅ Not Spam → *“This message is Safe”*  
- 🚨 Spam → *“Spam Message Detected!”*  

---

## 🛠️ Tech Stack & Tools  

- **Language**: Python 🐍  
- **Framework**: Streamlit 🚀  
- **ML & NLP**: Scikit-learn, NLTK, TF-IDF, SpaCy, TextBlob also trained on (Bag of words,word to vec)
- **Deep Learning (optional)**:Keras 
- **Data Handling & Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, WordCloud  
- **Model Evaluation**: Scikit-learn Metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)  
- **Version Control**: Git, GitHub  
- **Deployment**: Streamlit Community Cloud 
- **Environment Management**: pip, Virtualenv / venv (not included in github jupyter_env)
- **Others**: Pickle (Model Serialization), Joblib, Jupyter Notebook (Exploration)  


---

## 📂 Project Structure  
📩 Spam-Classifier/
│── spam-classification.py # Streamlit app for deployment and frontend
│── model.pkl # Trained ML model (Multinomial NB)
│── CountVectorizer.pkl # Saved TF-IDF vectorizer
|── spam-classifier.ipynb # jupyter notebook for total training portion
│── spam.csv # Training dataset
│── requirements.txt # Dependencies
│── README.md # Documentation
│── nltk_data/ # NLTK resources for cloud deployment

---


## 💻 Run Locally
1. Clone the Repository

Open your terminal and run:
git clone https://github.com/dibyendu17122003/spam-classifier.git

Then navigate into the project folder:
cd spam-classifier

2. Create and Activate Virtual Environment
Create a virtual environment:python -m venv .env
  Activate the environment:
    For Mac/Linux: source .env/bin/activate
    For Windows: .env\Scripts\activate
3. Install Dependencies
Install all required packages using: pip install -r requirements.txt
4. Run the Application
Start the Streamlit app with: streamlit run spam-classification.py / python -m streamlit run spam-classification.py


## 📦 Requirements
 1.Python 3.8+
 2.Streamlit
 3.Scikit-learn
 4.NLTK
 5.Pandas, NumPy

---
 
## 📊 Dataset  
- **Name:** SMS Spam Collection Dataset  
- **Size:** ~5,574 labeled messages  
- **Labels:**  
  - 'ham' (legitimate) → 4,827 samples  
  - 'spam' (unwanted) → 747 samples


---
## 🧠 Machine Learning Pipeline  
'''mermaid'''
flowchart TD
    A[Raw Messages (Input Data)] --> B[Data Cleaning & Preprocessing]
    B --> C[Exploratory Data Analysis (EDA)]
    C --> D[Tokenization]
    D --> E[Stopword Removal & Stemming]
    E --> F[TF-IDF Vectorization / Feature Engineering]
    F --> G[Train-Test Split]
    G --> H[Model Training]
    H --> I[Model Improvement & Optimization]
    I --> J[Cross-Validation & Evaluation Metrics]
    J --> K[Hyperparameter Tuning (Optional)]
    K --> L[Save Model & Vectorizer as .pkl]
    L --> M[Streamlit App Integration]
    M --> N[Deployment on Streamlit / Render]
    N --> O[Monitoring & Maintenance]


---

## 📊 Future Improvements
 1.🔮 Add deep learning models (RNN, LSTM, Transformers)
 2.📬 Extend to multilingual spam detection
 3.📱 Build a mobile-ready interface
 4.📊 Deploy as a REST API service

---
 
## 👨‍💻 Author :--Dibyendu Karmahapatra
💡 “From raw data to deployment — every step crafted with precision.
Zero Spam. Maximum Inbox Clarity. Always ✨”
![GitHub Stars](https://img.shields.io/github/stars/dibyendu17122003/MOVIE-RECOMENDATION-SYSTEM-ML?style=social)
![GitHub Forks](https://img.shields.io/github/forks/dibyendu17122003/MOVIE-RECOMENDATION-SYSTEM-ML?style=social)








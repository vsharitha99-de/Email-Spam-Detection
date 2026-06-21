# 📧 Email Spam Detection System

A Machine Learning-based Email/SMS Spam Classifier that predicts whether a message is **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques and a trained classification model. The application is deployed using Streamlit for real-time predictions.

## 🚀 Features

* Classifies messages as Spam or Not Spam
* Text preprocessing and cleaning
* TF-IDF Vectorization for feature extraction
* Machine Learning model for prediction
* Interactive Streamlit web application
* Real-time message analysis

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Pickle

## 📂 Project Structure

Email-Spam-Detection/

├── app/

│ └── app.py

├── data/

│ └── spam.csv

├── models/

│ ├── model.pkl

│ └── vectorizer.pkl

├── src/

│ ├── preprocess.py

│ └── train.py

├── requirements.txt

└── README.md

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Text Vectorization using TF-IDF
4. Model Training
5. Model Evaluation
6. Real-Time Prediction using Streamlit

## ▶️ Installation

Clone the repository:

git clone https://github.com/vsharitha99-de/Email-Spam-Detection.git

Navigate to the project folder:

cd Email-Spam-Detection

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## 📈 Model Performance

The model is trained on SMS Spam Collection Dataset and evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score

## 📸 Application Preview

Add screenshots of your Streamlit application here.

## 🎯 Future Improvements

* Deep Learning based classification
* Email attachment analysis
* Multi-language spam detection
* Model deployment on cloud platforms

## 👩‍💻 Author

Haritha V

BCA Student | Python Developer | Data Analytics & Machine Learning Enthusiast


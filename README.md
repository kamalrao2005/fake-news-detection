# 📰 Fake News Detection using Machine Learning

A Machine Learning project that detects whether a news article is **Fake** or **Real** using **Natural Language Processing (NLP)** and **Logistic Regression**.

---

## 📌 Project Overview

Fake news has become a major challenge in today's digital world. This project uses Machine Learning and Natural Language Processing (NLP) techniques to classify news articles as **Fake** or **Real** based on their textual content.

The model is trained on a publicly available dataset using **TF-IDF Vectorization** and **Logistic Regression**, achieving an accuracy of approximately **98.27%**.

---

## 🚀 Features

- Detects Fake or Real news articles
- Text preprocessing using NLP
- TF-IDF Vectorization
- Logistic Regression Classifier
- High prediction accuracy (~98.27%)
- Interactive command-line prediction
- Confusion Matrix and Classification Report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorizer
- Logistic Regression

---

## 📂 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **Fake and Real News Dataset** available on Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains:

- Fake News Articles
- Real News Articles

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/kamalrao2005/fake-news-detection.git
```

Move into the project folder

```bash
cd fake-news-detection
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the application using

```bash
python main.py
```

Example

```
Enter a news article:

NASA successfully launched a new satellite.

Prediction:

Real News
```

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | 98.27% |
| Precision | 98% |
| Recall | 98% |
| F1 Score | 98% |

The model was evaluated using:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## 💡 Future Improvements

- Build a Streamlit Web Application
- Deploy the application online
- Save the trained model using Joblib
- Compare multiple ML algorithms
- Add Deep Learning models (LSTM/BERT)
- Improve text preprocessing
- Add confidence score for predictions

---

## 👨‍💻 Author

**Kamal Rao**

GitHub:
https://github.com/kamalrao2005

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Load Dataset
# -------------------------------

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle the dataset
data = data.sample(frac=1, random_state=42)

# Keep only required columns
data = data[["text", "label"]]

# -------------------------------
# Features and Labels
# -------------------------------

X = data["text"]
y = data["label"]

# -------------------------------
# Split Data
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Convert Text to Numbers
# -------------------------------

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# -------------------------------
# Train Model
# -------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Test with Custom News
# -------------------------------

while True:

    print("\nEnter a news article (or type quit):")
    news = input()

    if news.lower() == "quit":
        break

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("\nReal News")
    else:
        print("\nFake News")
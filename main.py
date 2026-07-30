import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
data = pd.concat([fake, true], ignore_index=True)

# Keep only required columns
data = data[["text", "label"]]

# Split data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# User prediction
while True:
    news = input("\nEnter News: ")

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    if result[0] == 1:
        print("✅ Real News")
    else:
        print("❌ Fake News")

    again = input("Check another? (y/n): ")

    if again.lower() != "y":
        break
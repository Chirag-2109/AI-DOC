import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data/bbc"   

texts = []
labels = []

for category in os.listdir(DATA_DIR):
    cat_path = os.path.join(DATA_DIR, category)

    if not os.path.isdir(cat_path):
        continue

    for file in os.listdir(cat_path):
        file_path = os.path.join(cat_path, file)

        with open(file_path, encoding="latin1") as f:
            texts.append(f.read())
            labels.append(category)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

joblib.dump(model, "models/doc_classifier.pkl")

print("Document classifier trained and saved.")

import joblib

model = joblib.load("models/doc_classifier.pkl")

text = """
the film industry announced a new film on comedy with jakers characeters
"""

print("Predicted category:", model.predict([text])[0])

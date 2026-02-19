import spacy

nlp = spacy.load("models/ner_model")

text = "John works at Microsoft and lives in New York."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)

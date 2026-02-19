import spacy
import pandas as pd
from spacy.training.example import Example
import random

# Load base model
nlp = spacy.load("en_core_web_sm")

# Read dataset
df = pd.read_csv("data/ner/ner_dataset.csv", encoding="latin1")

df = df.ffill()

sentences = []
entities = []
current_words = []
current_tags = []

for word, tag in zip(df["Word"], df["Tag"]):
    current_words.append(word)
    current_tags.append(tag)

    if word == ".":
        sentences.append(current_words)
        entities.append(current_tags)
        current_words = []
        current_tags = []

train_data = []

for words, tags in zip(sentences, entities):
    text = " ".join(words)
    ents = []
    start = 0

    for word, tag in zip(words, tags):
        end = start + len(word)

        if tag != "O":
            label = tag.split("-")[-1]
            ents.append((start, end, label))

        start = end + 1

    train_data.append((text, {"entities": ents}))

# Training
optimizer = nlp.resume_training()

for epoch in range(10):
    random.shuffle(train_data)
    losses = {}

    for text, annot in train_data[:2000]:   # use subset for speed
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        nlp.update([example], drop=0.3, losses=losses)

    print(f"Epoch {epoch} Loss: {losses}")

# Save model
nlp.to_disk("models/ner_model")
print("NER model saved.")

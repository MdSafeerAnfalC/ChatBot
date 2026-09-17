import json
import random
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents file
with open('ml interns/intents.json') as file:
    data = json.load(file)

stemmer = PorterStemmer()

# Prepare dataset
sentences = []
labels = []
label_map = {}

for i, intent in enumerate(data['intents']):
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])
    label_map[intent['tag']] = i

# Preprocess text
def tokenize_and_stem(text):
    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(text)]

vectorizer = CountVectorizer(tokenizer=tokenize_and_stem)
X = vectorizer.fit_transform(sentences)
y = np.array([label_map[label] for label in labels])

# Train ML model
model = LogisticRegression()
model.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    X_test = vectorizer.transform([user_input])
    pred = model.predict(X_test)[0]

    # Find intent tag
    intent_tag = list(label_map.keys())[list(label_map.values()).index(pred)]

    # Select random response
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])

# Chat loop
print("Chatbot is ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Bot:", chatbot_response(user_input))

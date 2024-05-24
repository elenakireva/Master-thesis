# Name: Elena Kireva
# Description: This Python code removes stop words from the data

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

nltk.download('stopwords')

# Load the stop words
stop_words = set(stopwords.words('english'))

# Remove stop words from text
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Load the json file (depending on which file I use)
with open('file.json', 'r') as file:
    data = json.load(file)

# Remove stop words from "combined_text" (again, depends which file)
for entry in data:
    entry['combined_text'] = remove_stopwords(entry['combined_text'])

# Write to a new json file (change the name of the file)
with open('new_file.json', 'w') as file:
    json.dump(data, file, indent=4)

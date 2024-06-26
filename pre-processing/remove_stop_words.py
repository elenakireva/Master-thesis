# Name: Elena Kireva
# Description: This Python code removes stop words from the data

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# load the json file (file with dementia and healthy)
with open('combined_withSW.json', 'r') as file:
    data = json.load(file)

for entry in data:
    entry['combined_text'] = remove_stopwords(entry['combined_text'])

with open('combined_withoutSW.json', 'w') as file:
    json.dump(data, file, indent=4)

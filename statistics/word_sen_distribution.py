# Name: Elena Kireva
# Description: This script returns the word and sentence distribution in the data (per condition)

import json
import re
from collections import defaultdict
import numpy as np

# json file with both dementia and healthy controls data
with open('combined.json', 'r') as f:
    data = json.load(f)

word_count = defaultdict(list)
sentence_length = defaultdict(list)
avg_sentence_length = defaultdict(list)
total_words = defaultdict(int)
total_sentences = defaultdict(list)

for entry in data:
    label = entry['condition']
    combined_text = entry['combined_text']
    sentences = re.split(r'[.!?]', combined_text)
    words = combined_text.split()

    num_words = len(words)
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
    num_sentences = len(sentence_lengths)

    word_count[label].append(num_words)
    sentence_length[label].extend(sentence_lengths)
    avg_sentence_length[label].append(np.mean(sentence_lengths))
    total_words[label] += num_words
    total_sentences[label].append(num_sentences)

def print_results(label, word_count, sentence_length, avg_sentence_length, total_words, total_sentences):
    print(f"\nLabel: {label}")
    print(f"Total Number of Words: {total_words[label]}")
    print(f"Word Count Range: {min(word_count[label])} - {max(word_count[label])}")
    print(f"Average Number of Words per Participant: {np.mean(word_count[label]):.2f}")
    print(f"Average Number of Sentences per Participant: {np.mean(total_sentences[label]):.2f}")
    print(f"Sentence Count Range: {min(total_sentences[label])} - {max(total_sentences[label])}")

for label in word_count:
    print_results(label, word_count, sentence_length, avg_sentence_length, total_words, total_sentences)


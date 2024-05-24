# Name: Elena Kireva
# Descrition: This python script calculates the average word per participant

import json

# Load the combined data (demetia and healthy)
with open('data.json', 'r') as file:
    combined_data = json.load(file)

# Create a dictionary to store the total number of words and occurrences of each condition
condition_word_counts = {}

# Calculate the total number of words and occurrences of each condition
for item in combined_data:
    conditions = item['condition'].split(', ')
    combined_text = item['combined_text']
    num_words = len(combined_text.split())
    for condition in conditions:
        if condition in condition_word_counts:
            condition_word_counts[condition]['total_words'] += num_words
            condition_word_counts[condition]['occurrences'] += 1
        else:
            condition_word_counts[condition] = {'total_words': num_words, 'occurrences': 1}

# Calculate the average number of words per condition
for condition, counts in condition_word_counts.items():
    average_words = counts['total_words'] / counts['occurrences']
    condition_word_counts[condition]['average_words'] = average_words

# Print the total and average number of words per condition
print("Total and Average Number of Words per Condition:")
for condition, counts in condition_word_counts.items():
    print(f"Condition: {condition}, Total Words: {counts['total_words']}, Average Words: {counts['average_words']}")

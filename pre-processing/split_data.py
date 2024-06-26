# Name: Elena Kireva
# Description: This python code splits the data into train, dev, test

import json
import random

def split_data(input_file, output_train, output_dev, output_test, proportions):
    with open(input_file, 'r') as f:
        data = json.load(f)

    dementia_data = [item for item in data if item['condition'] == 'dementia']
    healthy_data = [item for item in data if item['condition'] == 'healthy']

    # Shuffle the data for randomness
    random.shuffle(dementia_data)
    random.shuffle(healthy_data)

    # Calculate the number of samples for each split
    dementia_counts = [int(len(dementia_data) * p) for p in proportions]
    healthy_counts = [int(len(healthy_data) * p) for p in proportions]

    # Split the data
    train_data = dementia_data[:dementia_counts[0]] + healthy_data[:healthy_counts[0]]
    dev_data = dementia_data[dementia_counts[0]:dementia_counts[0]+dementia_counts[1]] + \
               healthy_data[healthy_counts[0]:healthy_counts[0]+healthy_counts[1]]
    test_data = dementia_data[-dementia_counts[2]:] + healthy_data[-healthy_counts[2]:]

    # Write the splits to output files
    with open(output_train, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(output_dev, 'w') as f:
        json.dump(dev_data, f, indent=4)
    
    with open(output_test, 'w') as f:
        json.dump(test_data, f, indent=4)

# Example usage
split_data('combined.json', 'train.json', 'dev.json', 'test.json', [0.8, 0.1, 0.1])


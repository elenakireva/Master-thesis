# Name: Elena Kireva
# Description: This python script creates a dataframe and a visualization for the demographic information of all participants, the graph is used in the methodology section

import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame
data = {
    'Category': ['Control', 'Control', 'Control', 'Control', 'Dementia', 'Dementia', 'Dementia', 'Dementia'],
    'Gender': ['Males', 'Females', 'Males', 'Females', 'Males', 'Females', 'Males', 'Females'],
    'Count': [89, 154, 117, 189, None, None, None, None],
    'Average Age': [64.9, 64.8, 69.5, 72.5, 64.8, 71.4, None, None],
}

df = pd.DataFrame(data)

# Fill NaN values for 'Count' column
df['Count'] = df['Count'].fillna(df.groupby('Category')['Count'].transform('mean'))

# Fill NaN values for 'Average Age' columns
df['Average Age'] = df['Average Age'].fillna(df.groupby('Category')['Average Age'].transform('mean'))

print(df)

# Data for plotting
categories = ['Control', 'Dementia']
metrics = ['Number of Subjects', 'Average Age']
num_males = [89, 117]
num_females = [154, 189]
avg_age_males = [64.9, 69.5]
avg_age_females = [64.9, 72.5]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.13
index = [0, 1]

# Plot bars for number of subjects
rects1 = ax.bar(index, num_males, bar_width, label='Males', color='skyblue')
rects2 = ax.bar([i + bar_width for i in index], num_females, bar_width, label='Females', color='pink')

# Plot bars for average age
rects3 = ax.bar([i + 2 * bar_width for i in index], avg_age_males, bar_width, label='Avg Age Males', color='steelblue')
rects4 = ax.bar([i + 3 * bar_width for i in index], avg_age_females, bar_width, label='Avg Age Females', color='palevioletred')

# Add text labels on each bar
def autolabel(rects, is_number_subjects=False):
    for rect in rects:
        height = rect.get_height()
        if is_number_subjects:
            ax.annotate('n={}'.format(int(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        else:
            ax.annotate('{:.1f}'.format(height), 
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

autolabel(rects1, is_number_subjects=True)
autolabel(rects2, is_number_subjects=True)
autolabel(rects3)
autolabel(rects4)

ax.set_xlabel('Condition')
ax.set_ylabel('Values')
ax.set_title('Distribution of age and gender')
ax.set_xticks([i + 2 * bar_width for i in index])
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()

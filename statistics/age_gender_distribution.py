# Name: Elena Kireva
# Description: This Python script takes .txt files with age/gender information (extracted from the original .cha files)
# and calculates the average age of the participants per condition and the gender distribution

import statistics

# read the text files (for the control group)
with open('participant_ages_control.txt', 'r') as file:
    lines_control = file.readlines()
    
# read the text files (for the dementia group)
with open('participant_ages_dementia.txt', 'r') as file:
    lines_dementia = file.readlines()

# convert them to integers
# for control group
ages_control = []
for line in lines_control:
    try:
        age = int(line.split()[0])
        ages_control.append(age)
    except (ValueError, IndexError):
        pass

# for dementia group
ages_dementia = []
for line in lines_dementia:
    try:
        age = int(line.split()[0])
        ages_dementia.append(age)
    except (ValueError, IndexError):
        pass

# average age (control)
if ages_control:
    average_age_control = sum(ages_control) / len(ages_control)
    print("average age for control:", average_age_control)
else:
    print("No valid age data found")

# average age (dementia)
if ages_dementia:
    average_age_dementia = sum(ages_dementia) / len(ages_dementia)
    print("average age for dementia:", average_age_dementia)
else:
    print("No valid age data found")
    
# variables for the gender count 
male_count_control = 0
male_count_dementia = 0
female_count_control = 0
female_count_dementia = 0
male_total_age_control = 0
female_total_age_control = 0
male_total_age_dementia = 0
female_total_age_dementia = 0

# gender count (control)
for line in lines_control:
    data = line.strip().split()
    if len(data) == 2:
        age_control = int(data[0])
        gender_control = data[1]
        if gender_control == "male":
            male_count_control += 1
            male_total_age_control += age_control
        elif gender_control == "female":
            female_count_control += 1
            female_total_age_control += age_control

# gender count (dementia)
for line in lines_dementia:
    data = line.strip().split()
    if len(data) == 2:
        age_dementia = int(data[0])
        gender_dementia = data[1]
        if gender_dementia == "male":
            male_count_dementia += 1
            male_total_age_dementia += age_dementia
        elif gender_dementia == "female":
            female_count_dementia += 1
            female_total_age_dementia += age_dementia

# calculate average age per gender and per condition
if male_count_control > 0:
    male_avg_age_control = male_total_age_control / male_count_control
else:
    male_avg_age_control = 0

if female_count_control > 0:
    female_avg_age_control = female_total_age_control / female_count_control
else:
    female_avg_age_control = 0
    
if male_count_dementia > 0:
    male_avg_age_dementia = male_total_age_dementia / male_count_dementia
else:
    male_avg_age_dementia = 0

if female_count_dementia > 0:
    female_avg_age_dementia = female_total_age_dementia / female_count_dementia
else:
    female_avg_age_dementia = 0

# results (control)
print("Number of males control:", male_count_control)
print("Number of females control:", female_count_control)
print("Average age of males control:", male_avg_age_control)
print("Average age of females control:", female_avg_age_control)

# results (dementia)
print("Number of males dementia:", male_count_dementia)
print("Number of females dementia:", female_count_dementia)
print("Average age of males dementia:", male_avg_age_dementia)
print("Average age of females dementia:", female_avg_age_dementia)

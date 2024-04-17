import pandas as pd

# Define the file paths for train, test, and gender submission CSV files
train_file_path = 'D:\Prodigy InfoTech Intern Projects/train.csv'
test_file_path = 'D:\Prodigy InfoTech Intern Projects/test.csv'
gender_submission_file_path = 'D:\Prodigy InfoTech Intern Projects/gender_submission.csv'

# Load the train, test, and gender submission CSV files using the specified file paths
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
gender_submission_df = pd.read_csv(gender_submission_file_path)

# Display the first few rows of each dataset to understand their structure
print("Train Data:")
print(train_df.head())

print("\nTest Data:")
print(test_df.head())

print("\nGender Submission Data:")
print(gender_submission_df.head())

# Fill missing age values with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Convert categorical variables (e.g., Sex, Embarked) into numerical representations
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Merge train and test datasets for analysis
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Display the updated combined dataset
print("Combined Data:")
print(combined_df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of passenger ages
plt.figure(figsize=(8, 6))
sns.histplot(combined_df['Age'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Passenger Ages')
plt.show()

# Visualize the survival rate by gender
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Survival Count by Gender')
plt.legend(['Not Survived', 'Survived'])
plt.show()
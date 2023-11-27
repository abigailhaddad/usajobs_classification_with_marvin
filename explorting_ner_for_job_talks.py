import pandas as pd
import ast
import numpy as np

# Function to convert string representation of lists into actual lists
def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return []

# Function to identify the elements not in label
def elements_not_in_label(programming_languages, software_tools, label):
    return list(set(programming_languages + software_tools) - set(label))

# Function to identify the elements in label but not in programming_languages and software_tools
def elements_not_in_programming_tools(programming_languages, software_tools, label):
    return list(set(label) - set(programming_languages + software_tools))

# Function to identify the overlap between label and programming_languages and software_tools
def overlap_elements(programming_languages, software_tools, label):
    return list(set(label) & set(programming_languages + software_tools))

# Function to calculate precision, recall, and F1 score
def calculate_metrics(df):
    tp = sum(len(row['overlap']) for index, row in df.iterrows())
    fp = sum(len(row['not_in_programming_tools']) for index, row in df.iterrows())
    fn = sum(len(row['not_in_label']) for index, row in df.iterrows())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def metrics_to_markdown(precision, recall, f1):
    return f"""
| Metric    | Value |
|-----------|-------|
| Precision | {precision:.4f} |
| Recall    | {recall:.4f} |
| F1 Score  | {f1:.4f} |
"""


# Read the sample data from CSV
df = pd.read_csv("../data/sample.csv")

# Apply the string to list conversion
df['programming_languages'] = df['programming_languages'].apply(string_to_list)
df['software_tools'] = df['software_tools'].apply(string_to_list)
df['label'] = df['label'].apply(string_to_list)

# Calculate the additional columns
df['not_in_label'] = df.apply(lambda row: elements_not_in_label(row['programming_languages'], row['software_tools'], row['label']), axis=1)
df['not_in_programming_tools'] = df.apply(lambda row: elements_not_in_programming_tools(row['programming_languages'], row['software_tools'], row['label']), axis=1)
df['overlap'] = df.apply(lambda row: overlap_elements(row['programming_languages'], row['software_tools'], row['label']), axis=1)

# Calculate the metrics
precision, recall, f1 = calculate_metrics(df)

# Print the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

metrics_md = metrics_to_markdown(precision, recall, f1)
with open('../data/metrics_and_confusion_matrix.md', 'w') as md_file:
    md_file.write(metrics_md)
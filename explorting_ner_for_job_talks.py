import pandas as pd
import numpy as np

# Function to check if either 'programming_languages' or 'software_tools' has any entries
def has_entities(row):
    return bool(row['programming_languages']) or bool(row['software_tools'])

df = final_aggregated_joa = pd.read_pickle("../data/final_aggregated_joa.pkl")

# Add a column to the DataFrame that indicates if the row has any entities
df['has_entities'] = df.apply(has_entities, axis=1)

# Separate the data into two DataFrames: one where entities are found and one where they are not
entities_df = df[df['has_entities']]
no_entities_df = df[~df['has_entities']]

# Determine the number of samples you want from each group
# Here we take more samples from rows with entities to ensure they are well-represented
num_samples_with_entities = min(25, len(entities_df))  # At most 25 or the number of available rows
num_samples_without_entities = 50 - num_samples_with_entities

# Ensure not to sample more than the available rows without entities
num_samples_without_entities = min(num_samples_without_entities, len(no_entities_df))

# Randomly sample from each group
sampled_with_entities = entities_df.sample(n=num_samples_with_entities, random_state=1)
sampled_without_entities = no_entities_df.sample(n=num_samples_without_entities, random_state=1)

# Combine the two samples into one DataFrame
sampled_df = pd.concat([sampled_with_entities, sampled_without_entities])

sampled_df  # This DataFrame now contains the sampled data


sampled_df.to_csv("../data/sampled_joa.csv")
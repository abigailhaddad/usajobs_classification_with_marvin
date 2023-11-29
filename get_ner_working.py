import pandas as pd
from transformers import pipeline

# Function to process a single text through the NER pipeline and return organized results
def get_ner_dict_for_text(text, ner_pipeline):
    # Initialize the NER pipeline
    

    # Process the text
    entities = ner_pipeline(text)

    # Organize entities into a dictionary
    label_word_dict = {}
    for entity in entities:
        label = entity['entity']
        word = entity['word']

        if label not in label_word_dict:
            label_word_dict[label] = []

        label_word_dict[label].append(word)

    # Remove duplicates by converting the lists to sets and back to lists
    label_word_dict = {label: list(set(words)) for label, words in label_word_dict.items()}

    return label_word_dict

# Function to apply NER to a pandas Series and return another Series with dictionaries
def apply_ner_to_series(series, model_name):
    return series.apply(lambda text: get_ner_dict_for_text(text, model_name))

# Example usage:

# Create a pandas DataFrame
df = pd.DataFrame({
    'texts': [
        "This job requires knowing Python, PowerBI, and R",
        "I love to use spreadsheets like Microsoft Excel.",
        "Her favorite libraries are PyTorch and NumPy"
    ]
})
ner_pipeline = pipeline("token-classification", model="numind/entity-recognition-multilingual-general-sota-v1")
# Apply NER to the 'texts' column
df['ner_results'] = apply_ner_to_series(df['texts'], ner_pipeline)

# Print the DataFrame
print(df)

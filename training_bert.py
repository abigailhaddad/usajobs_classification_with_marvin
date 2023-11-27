import pandas as pd
import ast
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('numind/generic-entity_recognition_NER-v1')


def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return []
    
def label_entities(row):
    labeled_entities = {}
    for entity in row['overlap']:
        if entity in row['programming_languages']:
            labeled_entities[entity] = 'PROGRAMMING_LANGUAGE'
        elif entity in row['software_tools']:
            labeled_entities[entity] = 'SOFTWARE'
    return labeled_entities

def verify_labels_in_text(df, text_column, labels_column):
    for index, row in df.iterrows():
        text = row[text_column]
        labels_dict = row[labels_column]
        for key in labels_dict.keys():
            if key not in text:
                print(f"Mismatch found in row {index}: '{key}' not in text.")

def get_word_to_offset_mapping(text):
    offset = 0
    word_to_offset = {}
    for word in text.split():
        offset = text.find(word, offset)
        word_to_offset[offset] = word
        offset += len(word)
    return word_to_offset

def tokenize_and_align_labels(text, labels_dict):
    tokenized_input = tokenizer(text, truncation=True, return_offsets_mapping=True)
    tokens = tokenized_input.tokens()
    offset_mapping = tokenized_input["offset_mapping"]

    word_to_offset = get_word_to_offset_mapping(text)

    print("Word to Offset Mapping:", word_to_offset)  # Debugging

    aligned_labels = []
    skip = 0  # Skip counter for multi-word entities

    for i, offset in enumerate(offset_mapping):
        if skip > 0:
            skip -= 1
            continue

        start, end = offset
        if start == end:  # Special token
            aligned_labels.append('O')
            continue

        matched_entity = None
        for entity, label in labels_dict.items():
            entity_start = text.find(entity)
            if entity_start == start:
                matched_entity = entity
                break

        if matched_entity:
            entity_tokens = tokenizer.tokenize(matched_entity)
            label = labels_dict[matched_entity]
            aligned_labels.append(f"B-{label}")
            aligned_labels.extend([f"I-{label}"] * (len(entity_tokens) - 1))
            skip = len(entity_tokens) - 1
        else:
            aligned_labels.append('O')

        print(f"Token: {tokens[i]}, Offset: {offset}, Matched Entity: {matched_entity}, Label: {aligned_labels[-1]}")  # Debugging

    return tokens, aligned_labels







df = pd.read_csv("../data/sample.csv")

df['overlap'] = df['overlap'].apply(string_to_list)
df['programming_languages'] = df['programming_languages'].apply(string_to_list)
df['software_tools'] = df['software_tools'].apply(string_to_list)

df['labeled_entities'] = df.apply(label_entities, axis=1)
verify_labels_in_text(df, 'duties_var', 'labeled_entities')


# Example usage
text_example = df.loc[1, 'duties_var']  # Replace with actual row access
labels_dict_example = df.loc[1, 'labeled_entities']  # Replace with actual row access
tokens, labels = tokenize_and_align_labels(text_example, labels_dict_example)
# Assuming tokenize_and_align_labels is already defined

# Tokenize and align labels for this specific row
# Inspect the tokens and labels
for token, label in zip(tokens, labels):
    if label!="O":
        print(f"{token}: {label}")


word_to_offset = get_word_to_offset_mapping(text_example)

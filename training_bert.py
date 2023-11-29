import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# Function to convert string representation of a list back to a list
def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return []

# Function to label entities as 'SOFTWARE'
def label_entities(row):
    labeled_entities = {}
    for entity in row['entities']:
        labeled_entities[entity] = 'SOFTWARE'
    return labeled_entities

# Function to verify if labels are present in the text
def verify_labels_in_text(df, text_column, labels_column):
    for index, row in df.iterrows():
        text = row[text_column]
        labels_dict = row[labels_column]
        for key in labels_dict.keys():
            if key not in text:
                print(f"Mismatch found in row {index}: '{key}' not in text.")

# Function to transform data into spaCy format
def transform_to_spacy_format(df, text_col, entity_col):
    spacy_data = []
    for _, row in df.iterrows():
        text = row[text_col]
        entity_info = row[entity_col]
        if not entity_info:
            continue
        entities = []
        for entity, label in entity_info.items():
            start = text.find(entity)
            if start != -1:
                end = start + len(entity)
                entities.append((start, end, label))
        if entities:
            spacy_data.append((text, {"entities": entities}))
    return spacy_data

# Function to create and return a DocBin from the training data
def create_docbin_for_training(train_data, model):
    nlp = spacy.blank(model)  # Create a blank Language object
    db = DocBin()  # Create a new DocBin
    for text, annot in tqdm(train_data):
        doc = nlp.make_doc(text)  # Create a Doc object from text
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print(f"Skipping entity in '{text}'")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db

# Function to evaluate the trained model
def evaluate_model(model_path, test_data):
    nlp = spacy.load(model_path)
    results = []
    for text, annot in test_data:
        doc = nlp(text)
        true_entities = [text[start:end] for start, end, _ in annot['entities']]
        pred_entities = [ent.text for ent in doc.ents]
        results.append({
            'text': text,
            'true_entities': true_entities,
            'predicted_entities': pred_entities
        })
    return results

# Main process
df = pd.read_csv("../data/1127.csv")
df['programming_languages'] = df['programming_languages'].apply(string_to_list)
df['software_tools_corrected'] = df['software_tools_corrected'].apply(string_to_list)
df['entities'] = df['programming_languages'] + df['software_tools_corrected'] 
df['labeled_entities'] = df.apply(label_entities, axis=1)
verify_labels_in_text(df, 'duties_var', 'labeled_entities')
df = df.drop_duplicates(subset='duties_var')
spacy_formatted_data = transform_to_spacy_format(df, 'duties_var', 'labeled_entities')

# Split the data
train_data, test_data = train_test_split(spacy_formatted_data, test_size=0.2)

# Create DocBin for training
db = create_docbin_for_training(train_data, "en")
db.to_disk("../train.spacy")  # Save the DocBin to disk

# After training, evaluate the model
# Replace with your model path after training
model_path = "./output/model-best"
evaluation_results = evaluate_model(model_path, test_data)

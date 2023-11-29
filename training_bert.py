import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import spacy
from spacy.util import minibatch, compounding
import random
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin


def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return []
    
def label_entities(row):
    labeled_entities = {}
    for entity in row['entities']:
        labeled_entities[entity] = 'SOFTWARE'
    return labeled_entities

def verify_labels_in_text(df, text_column, labels_column):
    for index, row in df.iterrows():
        text = row[text_column]
        labels_dict = row[labels_column]
        for key in labels_dict.keys():
            if key not in text:
                print(f"Mismatch found in row {index}: '{key}' not in text.")


def transform_to_spacy_format(df, text_col, entity_col):
    spacy_data = []

    for _, row in df.iterrows():
        text = row[text_col]
        entity_info = row[entity_col]

        # Skip rows with no entities
        if not entity_info:
            continue

        entities = []
        for entity, label in entity_info.items():
            # Find the start and end positions of each entity in the text
            start = text.find(entity)
            if start != -1:  # Entity found in the text
                end = start + len(entity)
                entities.append((start, end, label))

        if entities:
            spacy_data.append((text, {"entities": entities}))

    return spacy_data

def train_test_split_spacy(data, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

def train_spacy_model(train_data, model_output_path="../output/model"):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            nlp.get_pipe("ner").add_label(ent[2])

    # Train the model
    optimizer = nlp.begin_training()
    for itn in range(10):  # Adjust the number of iterations as needed
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, drop=0.5, losses=losses)
        print(f"Losses at iteration {itn}: {losses}")

    # Save the model
    nlp.to_disk(model_output_path)

def evaluate_model(model_path, test_data):
    nlp = spacy.load(model_path)
    correct_preds, total_preds, total_true = 0, 0, 0

    for text, annot in test_data:
        doc = nlp(text)
        true_ents = [(ent[0], ent[1], ent[2]) for ent in annot['entities']]
        pred_ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Comparing entities
        total_preds += len(pred_ents)
        total_true += len(true_ents)
        correct_preds += len(set(true_ents) & set(pred_ents))

    precision = correct_preds / total_preds if total_preds > 0 else 0
    recall = correct_preds / total_true if total_true > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


df = pd.read_csv("../data/1127.csv")

df['programming_languages'] = df['programming_languages'].apply(string_to_list)
df['software_tools_corrected'] = df['software_tools_corrected'].apply(string_to_list)
df['entities'] = df['programming_languages'] + df['software_tools_corrected'] 

df['labeled_entities'] = df.apply(label_entities, axis=1)
verify_labels_in_text(df, 'duties_var', 'labeled_entities')
 
df = df.drop_duplicates(subset= 'duties_var')

spacy_formatted_data = transform_to_spacy_format(df, 'duties_var', 'labeled_entities')
TRAIN_DATA= spacy_formatted_data


nlp = spacy.blank("en")

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("../train.spacy") # save the docbin object


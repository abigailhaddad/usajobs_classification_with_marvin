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

def evaluate_and_identify_errors(model_path, test_data):
    nlp = spacy.load(model_path)
    results = []

    for text, annot in test_data:
        doc = nlp(text)
        true_entities = set([text[start:end] for start, end, _ in annot['entities']])
        pred_entities = set([ent.text for ent in doc.ents])

        overlap = true_entities.intersection(pred_entities)
        false_positives = pred_entities.difference(true_entities)
        false_negatives = true_entities.difference(pred_entities)

        results.append({
            'text': text,
            'true_entities': true_entities,
            'predicted_entities': pred_entities,
            'overlap': overlap,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        })

    return results

def append_results_to_dataframe(original_df, results, train_data, test_data):
    # Create a set of test texts for easy checking
    test_texts = set([text for text, _ in test_data])

    # Add new columns to the DataFrame
    original_df['DataSet'] = original_df['duties_var'].apply(lambda x: 'Test' if x in test_texts else 'Train')
    original_df['Overlap'] = ''
    original_df['False_Positives'] = ''
    original_df['False_Negatives'] = ''

    # Map text to its corresponding row index
    text_to_index = {row['duties_var']: index for index, row in original_df.iterrows()}

    # Update the DataFrame with the results
    for result in results:
        index = text_to_index[result['text']]
        original_df.at[index, 'Overlap'] = list(result['overlap'])
        original_df.at[index, 'False_Positives'] = list(result['false_positives'])
        original_df.at[index, 'False_Negatives'] = list(result['false_negatives'])

    # Write the updated DataFrame to a new CSV file
    original_df.to_csv('updated_data.csv', index=False)

def append_results_to_dataframe(df, evaluation_results, train_data, test_data):
    # Extract just the entities from the training data for comparison
    train_entities = set()
    for _, annotations in train_data:
        for start, end, label in annotations["entities"]:
            train_entities.add(df['duties_var'][start:end])

    # Add new columns to the DataFrame
    df['DataSet'] = 'Train'  # Initialize all as 'Train'
    df['Overlap'] = None
    df['False_Positives'] = None
    df['False_Negatives'] = None

    # Update the DataFrame with the results
    for result in evaluation_results:
        text = result['text']
        true_entities = result['true_entities']
        predicted_entities = result['predicted_entities']
        overlap = true_entities.intersection(predicted_entities)
        false_positives = predicted_entities.difference(true_entities)
        false_negatives = true_entities.difference(predicted_entities)

        # Find the row index in the original DataFrame
        row_index = df.index[df['duties_var'] == text].tolist()[0]

        # Update the 'DataSet' column to 'Test' for test data
        if text in [td[0] for td in test_data]:
            df.at[row_index, 'DataSet'] = 'Test'

        # Check if each predicted entity was seen in the training data
        seen_in_train = {entity for entity in overlap if entity in train_entities}
        not_seen_in_train = overlap.difference(seen_in_train)

        # Update the DataFrame
        df.at[row_index, 'Overlap'] = list(seen_in_train)
        df.at[row_index, 'False_Positives'] = list(false_positives)
        df.at[row_index, 'False_Negatives'] = list(false_negatives)

    # Write the updated DataFrame to a new CSV file
    df.to_csv('updated_evaluation_results.csv', index=False)

def collect_unique_entities(data):
    unique_entities = set()
    for _, row in data.iterrows():
        entities = row['labeled_entities']
        unique_entities.update(entities.keys())
    return unique_entities

def analyze_test_entities(train_data, test_data, evaluation_results):
    # Extract entities from the training data
    train_entities = set()
    for text, annot in train_data:
        for start, end, label in annot["entities"]:
            train_entities.add(text[start:end])

    # Extract true entities from the test data
    test_entities_true = set()
    for text, annot in test_data:
        for start, end, label in annot["entities"]:
            test_entities_true.add(text[start:end])

    # Initialize sets for analysis
    test_entities_found = set()
    test_entities_missed = set()

    # Process evaluation results
    for result in evaluation_results:
        predicted_entities = set(result['predicted_entities'])
        true_entities = set(result['true_entities'])

        # Add found and missed entities
        test_entities_found.update(predicted_entities)
        test_entities_missed.update(true_entities - predicted_entities)

    # Compare entities
    found_in_train = test_entities_found.intersection(train_entities)
    not_found_in_train = test_entities_found.difference(train_entities)
    missed_entities = test_entities_true - test_entities_found

    print("Entities found in test data that were in training data:", found_in_train)
    print("Entities found in test data that were NOT in training data:", not_found_in_train)
    print("Missed entities in test data:", missed_entities)

# Usage



df = pd.read_csv("../data/1127.csv")
df['programming_languages'] = df['programming_languages'].apply(string_to_list)
df['software_tools_corrected'] = df['software_tools_corrected'].apply(string_to_list)
df['entities'] = df['programming_languages'] + df['software_tools_corrected'] 
df['labeled_entities'] = df.apply(label_entities, axis=1)

# Split the DataFrame into training and test DataFrames
train_df, test_df = train_test_split(df, test_size=0.5)

# Verify labels in text for both training and test sets
verify_labels_in_text(train_df, 'duties_var', 'labeled_entities')
verify_labels_in_text(test_df, 'duties_var', 'labeled_entities')

# Transform the training and test DataFrames into spaCy format separately
train_data = transform_to_spacy_format(train_df, 'duties_var', 'labeled_entities')
test_data = transform_to_spacy_format(test_df, 'duties_var', 'labeled_entities')

# Create DocBin for training data
db = create_docbin_for_training(train_data, "en")
db.to_disk("../train.spacy")  # Save the DocBin to disk

# After training, evaluate the model using the test data
model_path = "../train/output/model-best"  # Adjust the path as needed
evaluation_results_detailed = evaluate_and_identify_errors(model_path, test_data)

# Analyze which entities in the test data were seen or not seen in the training data
analyze_test_entities(train_data, test_data, evaluation_results)

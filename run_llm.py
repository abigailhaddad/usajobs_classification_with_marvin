# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:18:04 2023

@author: abiga
"""
import pandas as pd
from marvin_classes import JobAnalyzer, TitleContraster
from marvin import settings
import logging
import os
import time
import datetime



def load_data(file_path):
    return pd.read_pickle(file_path)

def process_batch(df_batch, batch_num):
    
    # 
    df_batch['analyzer'] = df_batch['duties_var'].apply(JobAnalyzer)
    df_batch['category'] = df_batch['analyzer'].apply(lambda x: x.category)
    df_batch['programming_languages'] = df_batch['analyzer'].apply(lambda x: x.programming_languages)
    df_batch['software_tools'] = df_batch['analyzer'].apply(lambda x: x.software_tools)
    df_batch['job_title'] = df_batch['analyzer'].apply(lambda x: x.job_title)

    
    # Run TitleContraster
    df_batch['title_contrast'] = df_batch.apply(lambda row: TitleContraster(row['job_title'], row['positionTitle']), axis=1)
    df_batch['mismatch_level'] = df_batch['title_contrast'].apply(lambda x: x.mismatch_level)

    # Drop temporary columns
    df_batch = df_batch.drop(columns=['analyzer', 'title_contrast'])

    return df_batch


def batch_processing(df, BATCH_SIZE, DIR_NAME, FILE_PREFIX):
    batch_num = 0
    while len(df) > 0:
        print(f"Processing batch {batch_num}, {len(df)} records left")
        
        # Start the timer here
        start_time = datetime.datetime.now()
        
        df_batch = df.sample(min(BATCH_SIZE, len(df)), random_state=42)
        df = df.drop(df_batch.index)

        if not os.path.exists(f"{DIR_NAME}/{FILE_PREFIX}{batch_num}.pkl"):
            processed_df = process_batch(df_batch, batch_num)
            save_batch(processed_df, batch_num, DIR_NAME, FILE_PREFIX)
            
        # Stop the timer
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).seconds / 60  # Convert to minutes
        
        # Calculate the number of batches left and predict the time left
        batches_left = (len(df) // BATCH_SIZE) + 1  # This ensures any leftover records also get a batch
        predicted_time_left = elapsed_time * batches_left
        
        print(f"Batch {batch_num} took {elapsed_time:.2f} minutes. Estimated time left: {predicted_time_left:.2f} minutes.")
        
        batch_num += 1

def save_batch(df, batch_num, DIR_NAME, FILE_PREFIX):
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)
    df.to_pickle(f"{DIR_NAME}/{FILE_PREFIX}{batch_num}.pkl")

def aggregate_batches(DIR_NAME, FILE_PREFIX):
    all_files = [f for f in os.listdir(DIR_NAME) if FILE_PREFIX in f]
    all_dfs = [pd.read_pickle(f"{DIR_NAME}/{f}") for f in all_files]

    aggregated_df = pd.concat(all_dfs, ignore_index=True)
    
    for f in all_files:
        os.remove(f"{DIR_NAME}/{f}")

    os.rmdir(DIR_NAME)
    return aggregated_df

def process_file(historical_file, file_with_llm_markings, sample_size=None, BATCH_SIZE=100, DIR_NAME="batched_files", FILE_PREFIX="batch_", max_retries=5, retry_delay=60):
    settings.llm_request_timeout_seconds = 6000
    input_path = f"../data/{historical_file}.pkl"
    output_path = f"../data/{file_with_llm_markings}.pkl"
    df = load_data(input_path)
    
    if sample_size:
        df = df.head(sample_size)
    
    retries = 0
    success = False
    while retries < max_retries and not success:
        try:
            batch_processing(df, BATCH_SIZE, DIR_NAME, FILE_PREFIX)
            success = True
        except Exception as e:
            retries += 1
            print(f"API failed, attempt {retries}/{max_retries}. Retrying in {retry_delay} seconds.")
            print(f"Error: {e}")
            time.sleep(retry_delay)  # wait for the specified delay before retrying

    if not success:
        print("Failed to process after maximum retries.")
        return None

    final_df = aggregate_batches(DIR_NAME, FILE_PREFIX)
    final_df.to_pickle(output_path)
    return final_df

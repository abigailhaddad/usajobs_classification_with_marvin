# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 09:30:05 2023

@author: abiga
"""

"""
A script to fetch historical job data from the USAJobs API and save it to a file.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
import numpy as np

def historical_search(position_series: str, start_date: str, end_date: str, page_size: int = 1000) -> pd.DataFrame:
    base_url = f"https://data.usajobs.gov/api/historicjoa?PositionSeries={position_series}&StartPositionOpenDate={start_date}&EndPositionOpenDate={end_date}"
    page_number = 1
    all_results = []

    while True:
        url = f"{base_url}&PageSize={page_size}&PageNumber={page_number}"
        print(f"Fetching: {url}")  # For debugging purposes
        response = requests.get(url, timeout=100)
        response_json = response.json()

        if "data" in response_json:
            job_data = response_json["data"]
            if not job_data:  # Exit loop if no more results are returned
                break
            all_results.extend(job_data)
        else:
            print("Warning: Unexpected JSON structure.")
            break  # Exit the loop if we don't find the expected structure

        # Pagination check
        if response_json["paging"]["metadata"]["currentPage"] >= response_json["paging"]["metadata"]["totalPages"]:
            break

        page_number += 1

    search_result_df = pd.DataFrame(all_results)
    return search_result_df



def pull_fields_from_dict(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Recursively unpacks all columns in a DataFrame containing dictionaries.

    Args:
        data_frame (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with all columns containing dictionaries unpacked.
    """
    data_frame_changed = True
    while data_frame_changed:
        data_frame_changed = False
        columns_to_remove = []
        new_columns = []

        for column in data_frame.columns:
            if dict in [type(item) for item in data_frame[column].values]:
                data_frame_changed = True
                new_columns.append(data_frame[column].apply(pd.Series))
                columns_to_remove.append(column)

        if data_frame_changed:
            data_frame = data_frame.drop(columns=columns_to_remove)
            data_frame = pd.concat([data_frame, *new_columns], axis=1)
            data_frame = data_frame.dropna(how='all', axis=1)

    return data_frame

def fetch_text_from_link(link_data):
    base_url = "https://data.usajobs.gov"
    
    # Check if the link_data has a href attribute and that 'rel' is 'AnnouncementText'
    if link_data and 'href' in link_data[0] and link_data[0]['rel'] == 'AnnouncementText':
        link = base_url + link_data[0]['href']
        response = requests.get(link, timeout=100)
        
        if response.status_code == 200:
            return response.text
    return None


def clean_html(value):
    if isinstance(value, str) and '<' in value:
        return BeautifulSoup(value, 'html.parser').get_text(separator=' ')
    return value



def fetch_historical(position_series: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Run a historical jobs search for the given parameters and return the data as a DataFrame.

    Args:
        position_series (str): The position series code.
        start_date (str): The start date for the job posting.
        end_date (str): The end date for the job posting.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched job data.
    """
    historical_data = historical_search(position_series, start_date, end_date)
    return historical_data

def expand_json(row):
    data_str = row['announcement_text']
    try:
        data_dict = json.loads(data_str)
        cleaned_data = {k: clean_html(v) for k, v in data_dict.items()}
        return pd.Series(cleaned_data)
    except:
        return pd.Series({})
    
    
def filter_duties(df: pd.DataFrame) -> pd.DataFrame:
    # Length based filtering
    min_length = 200
    length_based_filter = df['duties_var'].str.len() > min_length

    # Keyword/regex based filtering
    phrases_to_exclude = [
        "Click on the link",
        "To view the duties",
        "Please refer to",
        "Click on link provided in"
    ]
    contains_phrase = df['duties_var'].str.contains('|'.join(phrases_to_exclude), flags=re.IGNORECASE, regex=True)
    regex_filter = ~(contains_phrase.fillna(False))

    # Apply filters
    filtered_df = df[length_based_filter & regex_filter]
    
    # Excluded entries
    excluded_df = df[~(length_based_filter & regex_filter)]

    # Sorting based on length for results
    filtered_df = filtered_df.sort_values(by='duties_var', key=lambda col: col.str.len())
    excluded_df = excluded_df.sort_values(by='duties_var', key=lambda col: col.str.len(), ascending=False)
    
    # Printing desired results
    print("Shortest 5 that didn't get filtered out:\n")
    print(filtered_df['duties_var'].head(5).values)
    print("\nLongest 5 that got filtered out:\n")
    print(excluded_df['duties_var'].head(5).values)
    
    return filtered_df



def fetch_and_write_out_historical(start_date, end_date, file_name):
    # An example of fetching historical data for a given series and date range.
    position_series = "1560"
    historical_data_from_function = fetch_historical(position_series, start_date, end_date)   
    historical_data_from_function['announcement_text'] = historical_data_from_function['_links'].apply(fetch_text_from_link)
    expanded_data = historical_data_from_function.apply(expand_json, axis=1)
    result_df = pd.concat([historical_data_from_function, expanded_data], axis=1)
    result_df.to_pickle(f"../data/{file_name}_unfiltered.pkl")
    result_df['duties_var'] = np.where(result_df['duties'].notna(), result_df['duties'], result_df['majorDutiesList'])
    sorted_df = result_df.sort_values(by='duties_var', key=lambda x: x.str.len())
    filtered_df = filter_duties(sorted_df)
    dropped_count, total_count = len(sorted_df)-len(filtered_df), len(result_df)
    print(f'Filtering for length and phrase to exclude announcements without duty descriptions dropped {dropped_count} out of {total_count} rows')
    filtered_df.to_pickle(f"../data/{file_name}.pkl")




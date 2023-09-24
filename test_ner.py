# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:57:20 2023

@author: abiga
"""

def all_strings_present(row):
    duties = row['duties_var'].lower()
    
    # Create a single list of tools by merging 'programming_languages' and 'software_tools'
    combined_tools = row['programming_languages'] + row['software_tools']
    
    return all(tool.lower() in duties for tool in combined_tools)


def analyze_tool_discrepancies(df):
    # Helper function to calculate discrepancies for a single row
    def get_discrepancies(row):
        found_tools = set([tool.lower() for tool in row['found_tools']])
        combined_others = set([tool.lower() for tool in row['software_tools'] + row['programming_languages']])

        only_in_found_tools = found_tools - combined_others
        only_in_software_and_languages = combined_others - found_tools

        discrepancies = {}
        
        if only_in_found_tools:
            discrepancies["only_in_found_tools"] = list(only_in_found_tools)
        if only_in_software_and_languages:
            discrepancies["only_in_software_and_languages"] = list(only_in_software_and_languages)
            
        return discrepancies or None

    df['discrepancies'] = df.apply(get_discrepancies, axis=1)
    return df


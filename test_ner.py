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

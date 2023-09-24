# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:40:49 2023

@author: abiga
"""

# it's actually a data engineer!

df_llm.loc[(df_llm['positionTitle'] == "Data Scientist") & (df_llm['job_title'] == "Data Engineer")]

# super short duties section

df_llm['duties_length'] = df_llm['duties_var'].str.len()
df_sorted = df_llm.sort_values(by='duties_length')
df_sorted['usajobsControlNumber'].head()

# TPM 

df_llm.loc[df_llm['job_title']=="Technical Program Manager"]

# Analyst position

df_sorted.loc[(df_sorted['positionTitle'] == "Data Scientist") & (df_sorted['job_title'] == "Data Analyst")].iloc[20]



df_llm.loc[df_llm['positionTitle']=="Data Scientist"]['job_title'].value_counts(normalize=True).head()
    
    
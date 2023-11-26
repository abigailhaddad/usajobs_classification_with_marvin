# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:02:05 2023

@author: abiga
"""

from run_llm import process_file
from pull_historical import fetch_and_write_out_historical
from test_ner import all_strings_present, analyze_tool_discrepancies
from ner_with_regex import find_tools_in_duties
from marvin import settings

class Config:
    def __init__(self):
        # General configuration
        self.historical_file = "historical_joa"
        self.file_with_llm_markings = "final_aggregated_joa_1126"

        # Configuration for historical data
        self.historical = self.HistoricalConfig()

        # Configuration for LLM processing
        self.llm = self.LLMConfig()

    class HistoricalConfig:
        def __init__(self):
            self.start_date = "11/01/2023"
            self.end_date = "11/26/2023"

    class LLMConfig:
        def __init__(self):
            self.BATCH_SIZE = 10
            self.DIR_NAME = "batched_files"
            self.FILE_PREFIX = "batch_"
            self.sample_size = None

if __name__ == "__main__":
    print(settings)
    config = Config()
    
    df_historical = fetch_and_write_out_historical(config.historical.start_date, config.historical.end_date, config.historical_file)
    df_llm = process_file(config.historical_file, config.file_with_llm_markings, sample_size=config.llm.sample_size, BATCH_SIZE=config.llm.BATCH_SIZE, DIR_NAME=config.llm.DIR_NAME, FILE_PREFIX=config.llm.FILE_PREFIX)
    # this is now just doing occs unique to job_title, not any found in official ones

    
    df_llm['all_present'] = df_llm.apply(all_strings_present, axis=1)
    
    df_llm = find_tools_in_duties(df_llm)
    
    df_llm  = analyze_tool_discrepancies(df_llm)
    
    
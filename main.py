# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:02:05 2023

@author: abiga
"""

from run_llm import process_file
from pull_historical import fetch_and_write_out_historical
from graphics_frequency_job_title import frequencies_wordcloud
from clustering import summarize_data
from test_ner import all_strings_present, analyze_tool_discrepancies
from ner_with_regex import find_tools_in_duties
from marvin import settings

class Config:
    def __init__(self):
        # General configuration
        self.historical_file = "historical_joa"
        self.file_with_llm_markings = "final_aggregated_joa"
        self.wordcloud_name = "wordcloud"

        # Configuration for historical data
        self.historical = self.HistoricalConfig()

        # Configuration for LLM processing
        self.llm = self.LLMConfig()

    class HistoricalConfig:
        def __init__(self):
            self.start_date = "1/1/2022"
            self.end_date = "9/30/2023"

    class LLMConfig:
        def __init__(self):
            self.BATCH_SIZE = 10
            self.DIR_NAME = "batched_files"
            self.FILE_PREFIX = "batch_"
            self.sample_size = None


    
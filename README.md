# USAJobs 1560 Listings Analysis with LLM (Large Language Model)

This repository contains scripts and tools that are used to fetch historical listings of jobs from USAJobs with the occupational code 1560. After fetching the data, GPT 3.5 is employed to classify these job listings by title. Additionally, named entity recognition (NER) is utilized to extract names of tools and software from the job listings. The LLM processing, as well as the NER functionalities, are implemented using the Marvin package, which provides structured output from GPT 3.5.

## Core Functionalities:

- **Fetching Historical Data**: Using `pull_historical.py`, historical job listings from USAJobs with occupational code 1560 are fetched based on a specified date range.
- **Processing with LLM**: The `run_llm.py` script processes the fetched job listings using GPT 3.5 to classify them by title.
- **Visualization**: The `graphics_frequency_job_title.py` script creates a word cloud visualization to represent the frequency of job titles found in generated job titles but not the official job titles.
- **Data Summarization**: With the `clustering.py` script, the processed data can be summarized to get insights.
- **Named Entity Recognition (NER)**: The `test_ner.py` and `ner_with_regex.py` scripts are used to extract names of tools and software from the job listings.
- **Data Analysis**: The main script (`__main__` execution section) coordinates all steps and provides a structured analysis, including the creation of a CSV file showcasing the relationship between job titles and position titles.

## Configuration:

The configuration for the process, including file names and dates for fetching historical data, is handled through the `Config` class in the main script.

## Usage:

1. Ensure all dependencies are installed.
2. Modify the `Config` class if necessary to suit your requirements.
3. Run the main script to fetch, process, and analyze the USAJobs listings.

## Results:

The final results, including the classified job listings and the extracted tools and software names, are saved in structured formats for further analysis and visualization. A sample result includes the `title_fields.csv` that provides insights into the relationship between classified job titles and position titles.

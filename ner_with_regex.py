# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:15:44 2023

@author: abiga
"""
import re

def get_software_tools_and_languages():
    """Returns two lists of software tools and programming languages for data and software jobs:
       1. Tools that require strict boundary checks.
       2. Tools that don't.
    """

    bounded_tools = [
        'r', 'rust', 'git', 'aws', 'bash', 'java', 'perl', 'scala', 'conda', 'hive'
    ]

    unbounded_tools = [
        'python', 'c\+\+', 'c#', 'ruby', 'php', "power bi",
        'powershell', 'matlab', 'octave', 'kotlin', 'fortran', 'cobol', 'assembly',
        'kafka', 'prefect', 'spark', 'hadoop', 'pig', 'redshift', 'bigquery',
        'docker', 'kubernetes', 'ansible', 'terraform', 'jenkins', 'kibana', 'zookeeper',
        'grafana', 'prometheus', 'logstash', 'rabbitmq', 'selenium', 'nginx', 'tomcat',
        'mocha', 'gradle', 'maven', 'webpack', 'gulp', 'yarn', 'virtualenv',
        'sql', 'javascript', 'typescript', 'vba', 'shiny', 'r-shiny', 'rshiny',
        'azure', 'gcp', 'powerbi', 'tableau', 'qlikview', 'looker', 'd3\.js', 'postgresql',
        'mysql', 'mongodb', 'cassandra', 'sqlite', 'oracle', 'db2', 'elasticsearch', 'influxdb',
        'neo4j', 'arangodb', 'dynamodb', 'mssql', 'cockroachdb', 'riak', 'anaconda', 'jupyter',
        'rstudio', 'github', 'gitlab', 'pycharm', 'eclipse', 'intellij', 'vscode', 'atom',
        'netbeans', 'spyder', 'sublime', 'svn', 'mercurial', 'linux', 'unix', 'macos', 'iis',
        'pytest', 'npm', 'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'tensorflow', 'keras',
        'pytorch', 'seaborn', 'ggplot2', 'tidyverse', 'dplyr', 'tidyr', 'stringr', 'lubridate'
    ]

    return bounded_tools, unbounded_tools


def find_tools_in_duties(df):
    """Searches for software tools and programming languages in the 'duties_var' column of the DataFrame."""
    
    bounded_tools, unbounded_tools = get_software_tools_and_languages()

    def find_tools(text):
        found_bounded = [tool for tool in bounded_tools if re.search(r'\b' + tool + r'\b', text, re.IGNORECASE)]
        found_unbounded = [tool for tool in unbounded_tools if tool in text.lower()]
        
        # Debug prints
        if found_bounded:
            print(f"Found bounded tools: {found_bounded}")
        if found_unbounded:
            print(f"Found unbounded tools: {found_unbounded}")

        return found_bounded + found_unbounded

    df['found_tools'] = df['duties_var'].apply(find_tools)
    return df

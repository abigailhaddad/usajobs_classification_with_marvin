# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:15:02 2023

@author: abiga
"""
import os
from enum import Enum
import openai
from marvin import ai_classifier, ai_fn
import logging

#logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
#logging.disable(logging.CRITICAL) # if you want to turn off the logging

openai.api_key = os.environ.get("OPENAI_API_KEY")



@ai_classifier
class JobCategory(Enum):
    """Represents general areas of job responsibilities"""
    DATA_SCIENCE_ENGINEERING = "Data Science and Engineering"
    DATA_ANALYSIS_BI = "Data Analysis and Business Intelligence"
    RESEARCH_SCIENCE = "Research and Science"
    MANAGEMENT_LEADERSHIP = "Management and Leadership"
    
@ai_fn
def generate_job_title(duties: str) -> str:
    """Given `duties`, generates a specific job title based on the content."""

@ai_fn
def extract_entities(duties: str) -> list[str]:
    """
    The software tools and programming languages identified.
    """
    
class JobAnalyzer:
    def __init__(self, duties: str):
        self.duties = duties
        self.category = JobCategory(self.duties).value
        self.entities = extract_entities(self.duties)
        self.job_title = generate_job_title(self.duties)

    def __repr__(self):
        return f"Duties: {self.duties[:100]}...\nCategory: {self.category}\nClarity Score: {self.clarity:.2f}\nEntities: {self.entities}\nJob Title: {self.job_title}"

@ai_classifier
class JobTitleContrast(Enum):
    """
    Compare two job titles:
    
    If the titles are the same or one title is an obvious subset of the other, classify as "Similar".
    For example:
    - "Data Scientist" vs "Data Scientist, ZP-1560-II/III, BEA-MP-VFP" are considered similar because the core title is the same.
    - "Chief Data Officer" vs "Data Officer" are also similar.
    - "Data Analyst" vs "Business Intelligence Analyst" are similar.
    
    If the core roles or designations in the titles differ significantly, classify as "Different".
    For example:
    - "Data Analyst" vs "Data Scientist" are different as they represent distinct roles.
    """
    Similar = "Position titles are identical or extremely similar"
    Different = "Position titles are mismatched"


class TitleContraster:
    def __init__(self, job_title: str, official_title: str):
        self.job_title = job_title
        self.official_title = official_title
        
        # Create a combined string for classification
        combined_title_info = f"Generated Title: {self.job_title} | Official Title: {self.official_title}"
        
        # Classify the combined string using the JobTitleContrast classifier
        self.mismatch_level = JobTitleContrast(combined_title_info).value

    def __repr__(self):
        return f"Generated Job Title: {self.job_title}\nPosition Title: {self.official_title}\nMismatch Level: {self.mismatch_level}"



def test_functions():
    # List of duties for testing
    duties_samples_list = [
    # Mixed with software tools
    "Responsible for designing algorithms using Python. Experience with TensorFlow and PyTorch required. Collaborate using Jira and Git.",

    # Not fitting any labels
    "Handle office administration, schedule meetings, and manage office supplies.",

    # Explicit software mention
    "Develop data visualization dashboards using Tableau and PowerBI. Experience with SQL databases preferred.",

    # Another non-fitting description
    "Plan and execute marketing campaigns. Familiarity with Adobe Photoshop and Illustrator is a plus.",

    # Mixed duties
    "Work on ETL processes using Apache Kafka. Use R and Python for data analysis. Experience with Hadoop is essential."
    ]

    # Test with the provided JobAnalyzer class
    for duties in duties_samples_list:
        analysis = JobAnalyzer(duties)
        print(analysis)
        print("-" * 80)


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
    NONE_OF_THE_ABOVE = "None of the Above"


# AI function to generate a job title
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


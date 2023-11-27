# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:15:02 2023

@author: abiga
"""
import os
from enum import Enum
import openai
from marvin import settings, ai_model, settings
import logging
from typing import List
from pydantic import BaseModel

settings.llm_max_tokens=1500
llm_max_context_tokens=2500
settings.llm_temperature=0.0

#logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
#logging.disable(logging.CRITICAL) # if you want to turn off the logging

openai.api_key = os.environ.get("OPENAI_API_KEY")
settings.llm_model='openai/gpt-4'

@ai_model(instructions='''Extract programming languages and 
          named software tools from the given text. 
          Only return items directly supported in text.''')

class TechDetails(BaseModel):
    text: str
    programming_languages: List[str]
    software_tools: List[str]

class JobAnalyzer:
    def __init__(self, duties: str):
        self.duties = duties
        # Extract tech details using the TechDetails model
        tech_details = TechDetails(self.duties)
        self.programming_languages = tech_details.programming_languages
        self.software_tools = tech_details.software_tools

    def __repr__(self):
        return (f"Duties: {self.duties}\n"
                f"Programming Languages: {self.programming_languages}\n"
                f"Software Tools: {self.software_tools}")


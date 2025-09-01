# -*- coding: utf-8 -*-
"""
agent.py

A multi-agent CrewAI system for disinformation analysis.
This script is designed to be imported as a module. It defines the tools,
agents, and tasks for the analysis crew.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
import torch # Import torch at the top
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Import these too

# Load environment variables from a .env file
load_dotenv()

# --- This part can remain global as it is stateless configuration ---
llm = LLM(
    api_key=os.getenv("API"),
    model="gemini/gemini-2.5-flash",
    temperature=0.1,
    top_p=0.9
)

# --- TOOL DEFINITIONS (No changes needed here) ---

class HeadlineSimilarityInput(BaseModel):
    headline: str = Field(description="The headline text of the article")
    body: str = Field(description="The body text of the article")

class HeadlineSimilarityTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "HeadlineSimilarity"
    description: str = "Computes semantic similarity between a headline and body text using a scientifically-backed sentence embedding model."
    args_schema: Type[BaseModel] = HeadlineSimilarityInput

    def _run(self, headline: str, body: str) -> str:
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        emb1 = model.encode(headline, normalize_embeddings=True)
        emb2 = model.encode(body, normalize_embeddings=True)
        sim = util.cos_sim(emb1, emb2)[0][0]
        return f"Semantic similarity score between headline and body: {sim:.4f}"


class EmotionalManipulationInput(BaseModel):
    text: str = Field(description="The text to analyze for emotional content")

class EmotionalManipulationTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "EmotionalManipulationMeter"
    description: str = "Classifies a wide range of emotions in text using a model trained on the robust GoEmotions dataset."
    args_schema: Type[BaseModel] = EmotionalManipulationInput

    def _run(self, text: str) -> str:
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", max_length=512,truncation=True,return_all_scores=True)
        results = classifier(text)
        return f"Detected Emotions: {results}"


class BiasDetectionInput(BaseModel):
    text: str = Field(description="The text to analyze for media bias")

class BiasDetectionTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "BiasDetectionAnalyzer"
    description: str = "Detects media bias using a model specifically trained on the Media Bias in Content (MBIC) dataset."
    args_schema: Type[BaseModel] = BiasDetectionInput

    def _run(self, text: str) -> str:
      tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
      model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
      inputs = tokenizer(text, return_tensors="pt", truncation=False)
      head_len = 128
      tail_len = 382
      max_len = head_len + tail_len
      if inputs['input_ids'].shape[1] > max_len:
          input_ids = inputs['input_ids'][0]
          attention_mask = inputs['attention_mask'][0]
          token_type_ids = inputs['token_type_ids'][0]
          head_input_ids = input_ids[:head_len]
          head_attention_mask = attention_mask[:head_len]
          head_token_type_ids = token_type_ids[:head_len]
          tail_input_ids = input_ids[-tail_len:]
          tail_attention_mask = attention_mask[-tail_len:]
          tail_token_type_ids = token_type_ids[-tail_len:]
          concatenated_ids = torch.cat([head_input_ids, tail_input_ids])
          concatenated_mask = torch.cat([head_attention_mask, tail_attention_mask])
          concatenated_type_ids = torch.cat([head_token_type_ids, tail_token_type_ids])
          inputs['input_ids'] = concatenated_ids.unsqueeze(0)
          inputs['attention_mask'] = concatenated_mask.unsqueeze(0)
          inputs['token_type_ids'] = concatenated_type_ids.unsqueeze(0)
      labels = torch.tensor([0])
      outputs = model(**inputs, labels=labels)
      loss, logits = outputs[:2]
      return logits.softmax(dim=-1)[0].tolist()


class HtmlParsingInput(BaseModel):
    html: str = Field(description="The HTML content to parse")

class HtmlParsingTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "HtmlParser"
    description: str = "Parses HTML using Newspaper3k to extract headline, body, authors, and publish date."
    args_schema: Type[BaseModel] = HtmlParsingInput

    def _run(self, html: str) -> Dict[str, str]:
        article = Article('')
        article.set_html(html)
        article.parse()
        return {
            'headline': article.title,
            'body': article.text,
            'authors': ', '.join(article.authors),
            'publish_date': str(article.publish_date)
        }

# --- STANDALONE PARSING FUNCTION (No changes needed here) ---
def parse_html_content_without_llm(html_content: str) -> Dict[str, str]:
    # ... (no changes in this function) ...
    print("--- [Pre-Processing] Starting HTML parsing from content ---")
    try:
        html_parser = HtmlParsingTool()
        parsed_data = html_parser._run(html=html_content)
        print("--- [Pre-Processing] HTML Parsing Complete ---")
        return parsed_data
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None


# ===================================================================
# --- NEW CREW FACTORY FUNCTION ---
# This function creates and returns a new crew instance every time it's called.
# ===================================================================
def create_disinfo_crew():
    """
    Factory function to create a new instance of the disinformation analysis crew.
    This ensures each API request gets its own isolated crew, preventing state conflicts.
    """
    # --- AGENT DEFINITIONS ---
    headline_agent = Agent(
        role='Headline Analyzer',
        goal='Analyze the semantic similarity between the article headline and its body content to check for clickbait or misrepresentation.',
        backstory='An expert in Natural Language Understanding and semantic analysis, skilled at identifying discrepancies in text.',
        tools=[HeadlineSimilarityTool()],
        llm=llm,
        verbose=True
    )

    emotion_agent = Agent(
        role='Emotional Manipulation Analyst',
        goal='Detect the use of emotionally charged language in the text that could be intended to manipulate the reader.',
        backstory='A psychologist specializing in computational linguistics, trained to identify manipulative language patterns.',
        tools=[EmotionalManipulationTool()],
        llm=llm,
        verbose=True
    )

    bias_agent = Agent(
        role='Bias Detection Analyst',
        goal='Analyze the text for signs of media bias, such as loaded language, one-sided framing, or unbalanced reporting.',
        backstory='A seasoned journalist and media critic with expertise in identifying subtle and overt biases in news content.',
        tools=[BiasDetectionTool()],
        llm=llm,
        verbose=True
    )

    medical_agent = Agent(
        role='Medical detector',
        goal='To detect if text contains conspiracy or disinformation.',
        backstory='A meticulous conspiracy theory detector with a deep understanding of science and vaccines.',
        llm=llm
    )

    # --- TASK DEFINITIONS ---
    headline_task = Task(
        description='1. Use the HeadlineSimilarity tool to get a semantic similarity score between the headline "{headline}" and the body "{body}".\n2. Based on the score, determine if the headline is clickbait or an accurate representation of the content.',
        agent=headline_agent,
        expected_output='A single sentence conclusion stating whether the headline is considered clickbait and the similarity score it was based on. Example: "The headline is likely clickbait, with a low similarity score of 0.45."'
    )

    emotion_task = Task(
        description='Analyze the text "{body}" using the EmotionalManipulationMeter tool to identify the dominant emotions. The tool will provide a list of many emotions and their scores.',
        agent=emotion_agent,
        expected_output='A single sentence identifying the top one or two dominant emotions from the tool\'s output and an overall assessment of the emotional tone. Example: "The text is primarily driven by anger and disgust, creating a highly manipulative tone."'
    )

    bias_task = Task(
        description='Use the BiasDetectionAnalyzer tool on the text "{body}". The tool will return three probabilities for [left, center, right] bias.',
        agent=bias_agent,
        expected_output='Based on the highest probability from the tool, provide the final classification. Your response MUST BE a: Left-Wing, Center, or Right-Wing.'
    )

    medical_task = Task(
        description='Analyze the following text: "{body}". Your sole task is to identify if it contains conspiracy theories about Vaccines, GMOs, or 5G. Do not provide any explanation, reasoning, or thought processes. Your output must be a single, concise sentence.',
        agent=medical_agent,
        async_execution=False,
        expected_output='A single sentence stating which of the topics (Vaccines, GMOs, 5G) were detected as being part of a conspiracy theory. For example: "A conspiracy theory related to Vaccines was detected." If none are found, your response must be "No common medical conspiracy topics were detected."'
    )

    # --- SETUP AND RETURN THE NEW CREW ---
    return Crew(
        agents=[headline_agent, emotion_agent, bias_agent, medical_agent],
        tasks=[headline_task, emotion_task, bias_task, medical_task],
        process=Process.sequential,
        verbose=True
    )
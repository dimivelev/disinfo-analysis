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
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
import pandas as pd
from newspaper import Article
from typing import Dict, Any, Type
from pydantic import BaseModel, Field

# Load environment variables from a .env file
load_dotenv()



# Make sure GOOGLE_API_KEY is set in your .env file
llm = LLM(
    api_key=os.getenv("API"),
    model="gemini/gemini-2.5-flash",
    temperature=0.1,
    top_p=0.9
    
)


# --- TOOL DEFINITIONS ---

class HeadlineSimilarityInput(BaseModel):
    headline: str = Field(description="The headline text of the article")
    body: str = Field(description="The body text of the article")

class HeadlineSimilarityTool(BaseTool):
    name: str = "HeadlineSimilarity"
    description: str = "Computes semantic similarity between a headline and body text using a scientifically-backed sentence embedding model."
    args_schema: Type[BaseModel] = HeadlineSimilarityInput

    def _run(self, headline: str, body: str) -> str:
        # Recommended Model: BAAI/bge-large-en-v1.5 is a top performer on the MTEB benchmark.
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        emb1 = model.encode(headline, normalize_embeddings=True)
        emb2 = model.encode(body, normalize_embeddings=True)
        sim = util.cos_sim(emb1, emb2)[0][0]
        return f"Semantic similarity score between headline and body: {sim:.4f}"



class EmotionalManipulationInput(BaseModel):
    text: str = Field(description="The text to analyze for emotional content")

class EmotionalManipulationTool(BaseTool):
    name: str = "EmotionalManipulationMeter"
    description: str = "Classifies a wide range of emotions in text using a model trained on the robust GoEmotions dataset."
    args_schema: Type[BaseModel] = EmotionalManipulationInput

    def _run(self, text: str) -> str:
        # Recommended Model: Trained on Google's GoEmotions dataset with 27 emotion categories.
        from transformers import pipeline
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", max_length=512,truncation=True,return_all_scores=True)


        results = classifier(text)
        return f"Detected Emotions: {results}"
'''
class LogicalFallacyInput(BaseModel):
    text: str = Field(description="The text to analyze for logical fallacies")

class LogicalFallacyTool(BaseTool):
    name: str = "LogicalFallacyDetector"
    description: str = "Detects logical fallacies in text and attempts to contextualize claims with Wikidata."
    args_schema: Type[BaseModel] = LogicalFallacyInput

    def _run(self, text: str) -> str:
        # Recommended Model: DeBERTa is a strong architecture for complex reasoning tasks.
        fallacy_classifier = pipeline("text-classification", model="hegelai/deberta-v3-base-logical-fallacies")
        results = fallacy_classifier(text)
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
        context = "No specific entities found for immediate Wikidata lookup."
        if entities:
            try:
                entity_obj = wikidata_client.get(entities[0], load=True)
                context = f"Context for '{entities[0]}': {entity_obj.description}"
            except Exception:
                context = f"No direct Wikidata context found for '{entities[0]}'."
        return f"Detected Fallacies: {results}, Context: {context}"
'''


class BiasDetectionInput(BaseModel):
    text: str = Field(description="The text to analyze for media bias")

class BiasDetectionTool(BaseTool):
    name: str = "BiasDetectionAnalyzer"
    description: str = "Detects media bias using a model specifically trained on the Media Bias in Content (MBIC) dataset."
    args_schema: Type[BaseModel] = BiasDetectionInput

    def _run(self, text: str) -> str:
      from transformers import AutoTokenizer, AutoModelForSequenceClassification
      import torch

      tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
      model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

      # Tokenize the text without default truncation
      inputs = tokenizer(text, return_tensors="pt", truncation=False)

      # Define the head and tail lengths
      head_len = 128
      tail_len = 382
      max_len = head_len + tail_len

      # Check if the sequence is longer than our combined length
      if inputs['input_ids'].shape[1] > max_len:
          # Get the tokens for input_ids, attention_mask, and token_type_ids
          input_ids = inputs['input_ids'][0]
          attention_mask = inputs['attention_mask'][0]
          token_type_ids = inputs['token_type_ids'][0]

          # --- Start of The Fix ---

          # 1. Grab the head of each tensor
          head_input_ids = input_ids[:head_len]
          head_attention_mask = attention_mask[:head_len]
          head_token_type_ids = token_type_ids[:head_len]

          # 2. Grab the tail of each tensor
          tail_input_ids = input_ids[-tail_len:]
          tail_attention_mask = attention_mask[-tail_len:]
          tail_token_type_ids = token_type_ids[-tail_len:]

          # 3. Concatenate the head and tail for each tensor
          concatenated_ids = torch.cat([head_input_ids, tail_input_ids])
          concatenated_mask = torch.cat([head_attention_mask, tail_attention_mask])
          concatenated_type_ids = torch.cat([head_token_type_ids, tail_token_type_ids])

          # 4. Update the inputs dictionary with the new truncated tensors
          #    We use .unsqueeze(0) to add the batch dimension back
          inputs['input_ids'] = concatenated_ids.unsqueeze(0)
          inputs['attention_mask'] = concatenated_mask.unsqueeze(0)
          inputs['token_type_ids'] = concatenated_type_ids.unsqueeze(0)

          # --- End of The Fix ---

      # Provide a dummy label to get the loss (optional)
      labels = torch.tensor([0])
      outputs = model(**inputs, labels=labels)
      loss, logits = outputs[:2]


      # [0] -> left
      # [1] -> center
      # [2] -> right
      return logits.softmax(dim=-1)[0].tolist()


class FactClaimInput(BaseModel):
    text: str = Field(description="The text from which to extract and verify claims")

class FactClaimTool(BaseTool):
    name: str = "FactClaimVerifier"
    description: str = "Checks if text contains conspiracy theory if it is non-PRCT means it doesn't."
    args_schema: Type[BaseModel] = FactClaimInput

    def _run(self, text: str) -> str:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained("erikbranmarino/CT-BERT-PRCT")
        model = AutoModelForSequenceClassification.from_pretrained("erikbranmarino/CT-BERT-PRCT")

# Prepare your text

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get predicted class (0: Non-PRCT, 1: PRCT)
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

        return f"Class: {'PRCT' if predicted_class == 1 else 'Non-PRCT'}"




class MultimediaInput(BaseModel):
    url: str = Field(description="A placeholder for a multimedia URL")

class MultimediaTool(BaseTool):
    name: str = "MultimediaScanner"
    description: str = "Scans multimedia metadata (placeholder function)."
    args_schema: Type[BaseModel] = MultimediaInput

    def _run(self, url: str) -> str:
        return "Multimedia analysis is a placeholder. No actual scan performed."





class HtmlParsingInput(BaseModel):
    html: str = Field(description="The HTML content to parse")

class HtmlParsingTool(BaseTool):
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
# --- STANDALONE PARSING FUNCTION ---
def parse_html_content_without_llm(html_content: str) -> Dict[str, str]:
    """
    Parses HTML content using the HtmlParsingTool without any LLM.
    This is a pre-processing step.
    """
    print("--- [Pre-Processing] Starting HTML parsing from content ---")
    try:
        html_parser = HtmlParsingTool()
        parsed_data = html_parser._run(html=html_content)
        print("--- [Pre-Processing] HTML Parsing Complete ---")
        return parsed_data
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

# --- AGENT DEFINITIONS ---
headline_agent = Agent(
    role='Headline Analyzer',
    goal='Analyze the semantic similarity between the article headline and its body content to check for clickbait or misrepresentation.',
    backstory='An expert in Natural Language Understanding and semantic analysis, skilled at identifying discrepancies in text.',
    tools=[HeadlineSimilarityTool()],
    llm=llm,
    verbose=False
)

emotion_agent = Agent(
    role='Emotional Manipulation Analyst',
    goal='Detect the use of emotionally charged language in the text that could be intended to manipulate the reader.',
    backstory='A psychologist specializing in computational linguistics, trained to identify manipulative language patterns.',
    tools=[EmotionalManipulationTool()],
    llm=llm,
    verbose=False
)
'''
fallacy_agent = Agent(
    role='Logical Fallacy Detector',
    goal='Identify any logical fallacies present in the article\'s arguments.',
    backstory='A logician and debate champion with a knack for spotting flawed reasoning and formal fallacies in text.',
    tools=[LogicalFallacyTool()],
    llm=llm,
    verbose=True
)
'''
bias_agent = Agent(
    role='Bias Detection Analyst',
    goal='Analyze the text for signs of media bias, such as loaded language, one-sided framing, or unbalanced reporting.',
    backstory='A seasoned journalist and media critic with expertise in identifying subtle and overt biases in news content.',
    tools=[BiasDetectionTool()],
    llm=llm,
    verbose=False
)

fact_agent = Agent(
    role='Conspiracy theory detector',
    goal='To detect if text contains conspiracy or disinformation.',
    backstory='A meticulous conspiracy theory detector with a deep understanding of science and vaccines.',
    tools=[FactClaimTool()],
    llm=llm,
    verbose=False
)
medical_agent = Agent(
    role='Medical detector',
    goal='To detect if text contains conspiracy or disinformation.',
    backstory='A meticulous conspiracy theory detector with a deep understanding of science and vaccines.',
    llm=llm
)

# --- TASK DEFINITIONS ---
headline_task = Task(
    #description='Analyze the headline "{headline}" against the body of the article. Use the HeadlineSimilarity tool.',
    description='1. Use the HeadlineSimilarity tool to get a semantic similarity score between the headline "{headline}" and the body "{body}".\n2. Based on the score, determine if the headline is clickbait or an accurate representation of the content.',
    agent=headline_agent,
    async_execution=True,
    expected_output='A single sentence conclusion stating whether the headline is considered clickbait and the similarity score it was based on. Example: "The headline is likely clickbait, with a low similarity score of 0.45."'

)

emotion_task = Task(
    description='Analyze the text "{body}" using the EmotionalManipulationMeter tool to identify the dominant emotions. The tool will provide a list of many emotions and their scores.',
    agent=emotion_agent,
    async_execution=True,
    expected_output='A single sentence identifying the top one or two dominant emotions from the tool\'s output and an overall assessment of the emotional tone. Example: "The text is primarily driven by anger and disgust, creating a highly manipulative tone."'
)
'''
fallacy_task = Task(
    description='Scan the article body for logical fallacies: "{body}". Use the LogicalFallacyDetector tool.',
    agent=fallacy_agent,
    expected_output='A list of any detected logical fallacies and their scores, with a brief explanation.'
)
'''
bias_task = Task(
    description='Use the BiasDetectionAnalyzer tool on the text "{body}". The tool will return three probabilities for [left, center, right] bias.',
    agent=bias_agent,
    async_execution=True,
    expected_output='Based on the highest probability from the tool, provide the final classification. Your response MUST BE a: Left-Wing, Center, or Right-Wing.'
)
fact_task = Task(
    description='Use the FactClaimVerifier tool to analyze the text "{body}" for conspiratorial content. The tool will classify it as either "PRCT" (Potential Conspiracy Related Text) or "Non-PRCT".',
    agent=fact_agent,
    async_execution=True,
    expected_output='A single-sentence verdict based on the tool\'s output. If the tool returns "PRCT", state that the text likely contains conspiracy theories. If it returns "Non-PRCT", state that it does not.'
)
medical_task = Task(
    description='Analyze the following text: "{body}". Your sole task is to identify if it contains conspiracy theories about Vaccines, GMOs, or 5G. Do not provide any explanation, reasoning, or thought processes. Your output must be a single, concise sentence.',
    agent=medical_agent,
    async_execution=False,
    expected_output='A single sentence stating which of the topics (Vaccines, GMOs, 5G) were detected as being part of a conspiracy theory. For example: "A conspiracy theory related to Vaccines was detected." If none are found, your response must be "No common medical conspiracy topics were detected."'

)

# --- SETUP THE CREW ---
disinfo_crew = Crew(
    agents=[headline_agent, emotion_agent, bias_agent, fact_agent,medical_agent],
    tasks=[headline_task, emotion_task, bias_task, fact_task,medical_task],
    #agents=[medical_agent],
    #tasks=[medical_task],
    process=Process.sequential
     # Using verbose level 2 for detailed output
)
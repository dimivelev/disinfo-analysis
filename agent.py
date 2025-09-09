# -*- coding: utf-8 -*-
"""
agent.py

A multi-agent CrewAI system for disinformation analysis.
This script is designed to be imported as a module. It defines the tools,
agents, and tasks for the analysis crew.
"""
import torch.nn.functional as F

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
        #model = SentenceTransformer('christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection')
        #classifier = pipeline("text-classification", model="christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection", max_length=512,truncation=True,return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained('christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection')
        model = AutoModelForSequenceClassification.from_pretrained("christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection")
        
        encoded_input = tokenizer(headline, return_tensors='pt', max_length=512, truncation=True)

    # Perform the forward pass with no_grad to save memory and computation
        with torch.no_grad():
            output = model(**encoded_input)

    # The output contains the logits
        logits = output.logits

    # Apply softmax to the logits to get probabilities
        probabilities = F.softmax(logits, dim=1)

    # Get the probability of the predicted class
        predicted_probability, predicted_class_index = torch.max(probabilities, dim=1)

    # The model's config tells us the labels: id2label: {0: 'not clickbait', 1: 'clickbait'}
        predicted_label = model.config.id2label[predicted_class_index.item()]
    
    # Format the output string
        return f"Detected class: {predicted_label} (Probability: {predicted_probability.item():.2%})"



class EmotionalManipulationInput(BaseModel):
    text: str = Field(description="The text to analyze for emotional content")

class EmotionalManipulationTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "EmotionalManipulationMeter"
    description: str = "Classifies a wide range of emotions in text using a model trained on the robust GoEmotions dataset."
    args_schema: Type[BaseModel] = EmotionalManipulationInput

    def _run(self, text: str) -> str:
        classifier = pipeline(
            "text-classification", 
            model="cirimus/modernbert-base-go-emotions",
            return_all_scores=True
            )

        
        predictions = classifier(text)

# Print top 5 detected emotions
        sorted_preds = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
        top_5 = sorted_preds[:5]

        print("\nTop 5 emotions detected:")
        for pred in top_5:
            print(f"\t{pred['label']:10s} : {pred['score']:.3f}")
        return str(top_5)
class BiasDetectionInput(BaseModel):
    text: str = Field(description="The text to analyze for media bias")

class BiasDetectionTool(BaseTool):
    # ... (no changes in the tool's code) ...
    name: str = "BiasDetectionAnalyzer"
    description: str = "Detects media bias using a model specifically trained on the Media Bias in Content (MBIC) dataset."
    args_schema: Type[BaseModel] = BiasDetectionInput

    def _run(self, text: str) -> str:
# Replace "test0198/modernbert-political-bias" with your actual repository ID on the Hugging Face Hub
        model_name = "Faith1712/Political-bias-ModernBERT-base"

# Load the tokenizer and model from the Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Assuming you have a GPU available, move the model to the GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        
# Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=8000)

# Move the input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

# Get the model's output (logits)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

# Get the predicted class index
        predicted_class_id = torch.argmax(logits, dim=1).item()

# Map the predicted class index to a label (you'll need to know the mapping from your training data)
# For example, if 0=left, 1=center, 2=right, you would use a dictionary:
        label_map = {0: "Left", 1: "Center", 2: "Right"} # Replace with your actual label mapping
        predicted_label = label_map.get(predicted_class_id, "Unknown")

        return predicted_label


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
        description="""Analyze the following text: "{body}".


You are tasked with identifying the trait of conspiratorial thinking in a given piece of text.
You have the following traits to choose from: Contradictory, Overriding suspicion, Nefarious
intent, Persecuted victim, Immune to evidence, and Re-interpreting randomness.
Here are definitions of the traits and what to look for in each one:
Contradictory: Conspiracy theorists can simultaneously believe in ideas that are mutually
contradictory. For example, believing the theory that Princess Diana was murdered, while
also believing that she faked her own death. This is because the theorists’ commitment to
disbelieving the “official” account is so absolute, it doesn’t matter if their belief system is
incoherent. What to look for: The author expresses beliefs that are mutually exclusive. A
and B cannot both be true at the same time. But they express belief in both A and B as a
means to counter popular opinion/the official account.
Overriding suspicion: Conspiratorial thinking involves a nihilistic degree of skepticism towards the official account. This extreme degree of suspicion prevents belief in anything that
doesn’t fit into the conspiracy theory. What to look for: Extreme/illogical distrust for official
accounts, a nihilistic degree of skepticism (such as officials being deceptive, incompetent,
lacking information, willfully ignorant, etc), Synonyms for lies/lying – “Scamdemic”, or
gullibility, “sheep” not wanting to know the truth , ignorance due to complacency, sarcasm
or mocking of officials/institutions, quotations like “Climate scientist,” The global “crisis”
Nefarious intent: The motivations behind any presumed conspiracy are invariably assumed to
be nefarious. Conspiracy theories never propose that the presumed conspirators have benign
motivations. What to look for: mention of evil motivations such as greed, hatred, some kind
of indoctrination (i.e. a cult/nazism), lack of empathy, etc. The author’s interpretation might
be implicit or explicit.
Persecuted victim: Conspiracy theorists perceive and present themself as the victim of organized persecution. At the same time, they see themself as brave antagonists taking on the
villainous conspirators. Conspiratorial thinking involves a self-perception of simultaneously
being a victim and a hero. What to look for: the author paints themselves or their in-group
as the victim/the hero, the author identifies with the favorable side/the good guys in their
narrative. They use terms/words like us, the American people, non-snowflakes, etc.
Immune to evidence: Conspiracy theories are inherently self-sealing—evidence that counters a theory is re-interpreted as originating from the conspiracy. This reflects the belief that
the stronger the evidence against a conspiracy (e.g., the FBI exonerating a politician from
allegations of misusing a personal email server), the more the conspirators must want people
to believe their version of events (e.g., the FBI was part of the conspiracy to protect that
politician). What to look for: Evidence that undermines the conspiracy theory is repurposed
as part of the conspiracy. The author implies others are “in on it”. The author references/introduces evidence and refutes it by dismissal, alternative truth/science/evidence, common
sense, etc. They may also attack the source of the evidence. They may also deny commonly
accepted knowledge without directly referencing it
Re-interpreting randomness: The overriding suspicion found in conspiratorial thinking frequently results in the belief that nothing occurs by accident. Small random events, such as
intact windows in the Pentagon after the 9/11 attacks, are re-interpreted as being caused by
the conspiracy (because if an airliner had hit the Pentagon, then all windows would have
shattered and are woven into a broader, interconnected pattern. What to look for: The article
uses some event(s) and connects it/them to a larger conspiracy to support their narrative. Ask
yourself – Did the event(s) likely occur by chance? Did the event(s) likely occur independently of the conspiracy and/or each other?


Based on this logic, analyze the text. Your output must be a single, concise sentence. Do not provide any explanation, reasoning, or thought processes.
""",
        agent=medical_agent,
        async_execution=False,
        expected_output='A single sentence stating which of the topics (Vaccines, GMOs, 5G) were detected as being part of a conspiracy theory. For example: "A conspiracy theory related to Vaccines was detected." If none are found, your response must be "No common conspiracy topics were detected."'
    )

    # --- SETUP AND RETURN THE NEW CREW ---
    return Crew(
        agents=[headline_agent, emotion_agent, bias_agent, medical_agent],
        tasks=[headline_task, emotion_task, bias_task, medical_task],
        process=Process.sequential,
        verbose=True
    )
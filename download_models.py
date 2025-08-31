# download_models.py
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def download_all_models():
    """
    Downloads and caches all the required models from Hugging Face and Spacy.
    """
    print("--- Starting model downloads ---")

    # Models used in the application
    hugging_face_models = [
        'BAAI/bge-large-en-v1.5',
        'j-hartmann/emotion-english-distilroberta-base',
        'bucketresearch/politicalBiasBERT',
        'erikbranmarino/CT-BERT-PRCT',
        'bert-base-cased' # This is a dependency for politicalBiasBERT tokenizer
    ]

    spacy_model = "en_core_web_sm"

    # Download Hugging Face models
    for model_name in hugging_face_models:
        try:
            print(f"Downloading {model_name}...")
            if 'BAAI' in model_name:
                SentenceTransformer(model_name)
            elif 'pipeline' in globals() and 'j-hartmann' in model_name:
                 pipeline("text-classification", model=model_name, truncation=True)
            else:
                AutoTokenizer.from_pretrained(model_name)
                AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Could not download {model_name}. Reason: {e}")

    # Download Spacy model
    try:
        print(f"Downloading Spacy model: {spacy_model}...")
        spacy.cli.download(spacy_model)
        print(f"Successfully downloaded {spacy_model}")
    except Exception as e:
        print(f"Could not download {spacy_model}. Reason: {e}")

    print("--- All model downloads attempted ---")

if __name__ == "__main__":
    download_all_models()
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Datasets
SQUAD_DATASET = DATA_DIR / "train-v2.0.json"
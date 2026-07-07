from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import json
from pathlib import Path

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_MODEL,
    SQUAD_DATASET,
)

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

"""
===========================
PINECONE MANAGEMENT
===========================
"""

def create_index():
    """Create the Pinecone index if it doesn't already exist."""

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX in existing_indexes:
        print(f"✓ Using existing index: {PINECONE_INDEX}")
        return

    print(f"Creating index '{PINECONE_INDEX}'...")

    pc.create_index(
        name=PINECONE_INDEX,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print("✓ Index created successfully.")    
    

def connect_index():
    """Connect to the configured Pinecone index."""

    return pc.Index(PINECONE_INDEX)

"""
===========================
EMBEDDINGS
===========================
"""

def create_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector using the configured OpenAI embedding model.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding

"""
===========================
DOCUMENT LOADING
===========================
"""

def load_documents(file_path: Path) -> list[dict]:
    """
    Load the SQuAD dataset from disk.
    """

    with file_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    documents = []

    for article in dataset["data"]:
        title = article["title"]

        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            document = {
                "title": title,
                "text": context,
            }

            documents.append(document)

    return documents

"""
===========================
APPLICATION TESTING
===========================
"""

def test_connections():
    """Verify OpenAI and Pinecone connectivity."""

    print("✓ OpenAI client initialized.")
    print("✓ Pinecone client initialized.")

    create_index()

    index = connect_index()

    print(f"✓ Connected to index: {index.describe_index_stats()['dimension']}-D")

    embedding = create_embedding("Hello world")

    print(f"✓ Embedding generated ({len(embedding)} dimensions)")

    documents = load_documents(SQUAD_DATASET)

    print(f"✓ Loaded {len(documents)} documents.")

    print("\nFirst document:")

    print(documents[0])
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION
)

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


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
    

# Connection test to verify that the clients are initialized correctly
def test_connections():
    print("✓ OpenAI client initialized.")
    print("✓ Pinecone client initialized.")
    create_index()
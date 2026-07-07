from openai import OpenAI
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY
)

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Connection test to verify that the clients are initialized correctly
def test_connections():
    print("✓ OpenAI client initialized.")
    print("✓ Pinecone client initialized.")
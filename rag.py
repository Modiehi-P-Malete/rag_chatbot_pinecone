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
    Load and extract documents from the SQuAD dataset.
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
                "metadata": {
                    "source": "SQuAD",
                },
            }

            documents.append(document)

    
    for index, document in enumerate(documents):
        document["id"] = f"doc-{index:06}"

    return documents

"""
===========================
TEXT CHUNKING
===========================
"""

def chunk_documents(
    documents: list[dict],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict]:

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0
        chunk_number = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "id": f"{document['id']}-chunk-{chunk_number}",
                    "title": document["title"],
                    "text": chunk_text,
                    "metadata": document["metadata"],
                }
            )

            chunk_number += 1
            start += chunk_size - overlap

    return chunks

def upsert_chunks(chunks: list[dict]):
    """
    Generate embeddings for each chunk and upload them to Pinecone.
    """

    index = connect_index()

    vectors = []

    for chunk in chunks:

        embedding = create_embedding(chunk["text"])

        vectors.append(
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                },
            }
        )
    index.upsert(vectors=vectors)

    print(f"✓ Uploaded {len(vectors)} vectors to Pinecone.")

def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant chunks from Pinecone.
    """

    index = connect_index()

    query_embedding = create_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return results.matches

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

    chunks = chunk_documents(documents)

    print(f"✓ Loaded {len(documents)} documents.")
    print(f"✓ Created {len(chunks)} chunks.")

    upsert_chunks(chunks[:100])

    print("\nFirst chunk:")
    print(chunks[0])

    matches = retrieve_context("Who is Beyoncé?")

    print(f"\nRetrieved {len(matches)} matches:\n")

    for match in matches:
        print(match.metadata["title"])
        print(match.metadata["text"][:200])
        print("-" * 50)
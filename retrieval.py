"""
File for Hybrid Retrieval Pipeline
Query Processing: User query is converted into a vector.
Vector Similarity Search: The qdrant vector database is used to retrieve the top‐k most similar document chunks.
Graph Query Augmentation: Using the metadata from the vector search, Cypher queries are constructed to retrieve related nodes and relationships from Neo4j.
Merge Results: The context from the vector search and the graph query are merged to form a richer, contextually enhanced input for answer generation.
"""
from qdrant_client import models
from config import initialize_config
from dotenv import load_dotenv
import os
from mistralai import Mistral
import sys
import getpass
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter your MistralAI API key: ")
model = "mistral-small-latest"
client = Mistral(api_key=api_key)

qdrant_client, collection_name, neo4j_database, driver, embeddings = initialize_config()

async def retrieve(query):
    """Retrieve the most relevant chunks from the knowledge base"""
    # Get the embeddings for the query
    query_embedding = embeddings.embed_query(query)

    # Search the qdrant collection for the most relevant chunks and using the metadata from the vector search 
    # to construct Cypher queries to retrieve related nodes and relationships from Neo4j
    qdrant_results = await qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )

    # Getting the text of the most relevant chunks
    relevant_texts = []
    for result in qdrant_results:
        relevant_texts.append(result.payload["text"])
    all_relevant_texts = "\n".join(relevant_texts)

    # Constructing Cypher queries to retrieve related nodes and relationships from Neo4j
    async with driver.session(database=neo4j_database) as session:
        cypher = """
        MATCH (c:Chunk) 
        WHERE toLower(c.text) CONTAINS toLower($query)
        RETURN c.text AS text LIMIT 5

        """

        # Executing the Cypher queries
        neo4j_results = await session.run(cypher, {"query": query})
        cypher_texts = []
        async for record in neo4j_results:
            cypher_texts.append(record["text"])

    return all_relevant_texts, cypher_texts

async def generate_answer_with_mistral(query, milvus_context, neo4j_context, model="mistral-small-latest", temperature=0.3, max_tokens=300):
    """
    Generate an answer using the Mistral API based on the given query and contexts.
    
    Args:
        query (str): The user's question
        milvus_context (str): Context retrieved from Milvus/Qdrant
        neo4j_context (str): Context retrieved from Neo4j
        model (str): Mistral model to use
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens to generate
        
    Returns:
        str: The generated answer
    """
    # Create prompt with contexts
    system_prompt = """You are a helpful assistant that answers questions regarding a data science textbook."""
    
    user_prompt = f"""Based on the following contexts extracted from similar questions and graph data:
---
Vector DB context:
{milvus_context}
---
Neo4j context:
{neo4j_context}
---
Question:
{query}

Provide a concise, helpful answer."""
    
    # Call Mistral API
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    
    response = await client.chat.complete_async(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract and return the answer
    return response.choices[0].message.content.strip()





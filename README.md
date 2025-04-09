# Hybrid Retrieval Pipeline for Data Science Knowledge Base

## Overview
This project implements a Hybrid Retrieval Pipeline for querying a data science knowledge base. 
It combines vector similarity search using Qdrant and graph-based query augmentation using Neo4j to retrieve relevant information from a data science textbook and related concepts. 
The retrieved information is then used to generate answers to user queries using the Mistral-small-latest.

## Features
- Query Processing: Converts user queries into vector embeddings using a pre-trained embedding model.
- Vector Similarity Search: Uses Qdrant to retrieve the top-k most similar document chunks based on vector embeddings.
- Graph Query Augmentation: Constructs Cypher queries to retrieve related nodes and relationships from Neo4j, leveraging metadata from the vector search.
- Answer Generation: Combines the retrieved contexts and generates concise answers using the Mistral AI language model.
- Data Ingestion: Processes data science textbooks and ingests them into Qdrant and Neo4j for efficient retrieval.

## Project Structure

├── retrieval.py               # Main retrieval logic for hybrid pipeline
├── ingest.py                  # Script for ingesting data into Qdrant and Neo4j
├── knowledge_base.py          # Generates Cypher queries for Neo4j ingestion
├── knowledge_base.cypher      # Generated Cypher queries for Neo4j schema and data
├── data_science_concepts.json # JSON file containing chapters and generated concepts
├── ML-DS-TEXTBOOKS/           # Directory containing data science textbooks (PDFs)
├── config.py                  # Configuration for Qdrant, Neo4j, and embeddings
├── preprocess.py              # Preprocessing logic for text extraction
├── scripts.py                 # Contains funtions for running each file

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
    Create a .env file in the root directory with the following variables:
       QDRANT_URL=<your-qdrant-url>
       NEO4J_URI=<your-neo4j-uri>
       NEO4J_USERNAME=<your-neo4j-username>
       NEO4J_PASSWORD=<your-neo4j-password>
       NEO4J_DATABASE=<your-neo4j-database-name>
       MISTRAL_API_KEY=<your-mistral-api-key>

## How It Works
- Data Ingestion: Textbooks are preprocessed and split into chunks. Chunks are stored in Qdrant with their embeddings.
- Entity and Relationship Extraction: Concepts and relationships are created and stored in Neo4j. 
- Query Processing: User queries are converted into vector embeddings. Qdrant retrieves the top-k most similar chunks. Neo4j retrieves related nodes and relationships using Cypher queries.
- Answer Generation: The retrieved contexts are combined. Mistral AI generates a concise answer based on the combined context.

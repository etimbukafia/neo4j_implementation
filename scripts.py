import os
import asyncio
import sys

from preprocess import gather_data, preprocess_data, extract_entities_relationships, save_results
from knowledge_base import create_cypher_queries
from ingest import ingest_to_qdrant_and_neo4j, validate_environment
from retrieval import retrieve, generate_answer_with_mistral

# Extract entities and relationships
def extract_entities_relationships():
    # Gather data
    file_path = gather_data()
    text = preprocess_data(file_path)
    
    # Extract entities and relationships
    results = extract_entities_relationships(text)
    
    # Save results
    output_file = "data_science_concepts.json"
    save_results(results, output_file)

def create_knowledge_base():
    cypher = create_cypher_queries("data_science_concepts.json")
    print(cypher)

    # Saving the Cypher queries to a file
    with open("knowledge_base.cypher", "w", encoding="utf-8") as f:
        f.write(cypher)
    print("Cypher queries saved to knowledge_base.cypher")

async def ingest_data(PDF_DIR):
    validate_environment()
    file_path = gather_data()
    file_path = os.path.join(PDF_DIR, file_path)
    try:
        await ingest_to_qdrant_and_neo4j(file_path)
    except Exception as e:
        print(f"Error in ingestion: {e}")
   
    print("All files processed successfully.")

async def retrieve(query):
    milvus_context, neo4j_context = await retrieve(query)
    answer = await generate_answer_with_mistral(query, milvus_context, neo4j_context)
    return answer

if __name__ == "__main__":
    query = sys.argv[1]
    answer = asyncio.run(retrieve(query))
    print(answer)

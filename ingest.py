from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from preprocess import preprocess_data
from dotenv import load_dotenv
import uuid
import json
from config import initialize_config

load_dotenv()

def validate_environment():
    required_vars = ["QDRANT_URL", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return True


PDF_DIR = "ML-DS-TEXTBOOKS"

qdrant_client, collection_name, neo4j_database, driver, embeddings = initialize_config()

def create_chunks(file_path):
    """Create chunks from the text in the pdfs"""
    text = preprocess_data(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    processed_chunks = [{"text": chunk, "metadata": {"source": file_path}} for chunk in chunks]
    return processed_chunks


def create_concept_list():
    with open("data_science_concepts.json", "r") as file:
        data = json.load(file)
    all_concepts = []
    for chapter in data:
        all_concepts.extend(chapter["concepts"])
    cleaned_concepts = []
    for concept in all_concepts:
        if any(char in concept for char in "`[](){}\"'"):
            concept = concept.replace("'", "")
            print(f"Warning: Concept '{concept}' contains special characters")
        cleaned_concepts.append(concept)
    unique_concepts = list(set(cleaned_concepts))
    unique_concepts.sort()
    return unique_concepts

CONCEPTS_IN_GRAPH = create_concept_list()

async def ingest_to_qdrant_and_neo4j(file_path):
    """Ingest the embeddings into qdrant and neo4j"""
    chunks = create_chunks(file_path)
    if not chunks:
        print("No chunks to ingest.")
        return
    
    # Generate all embedding IDs and prepare data
    all_data = []
    for i, chunk in enumerate(chunks):
        embeddingsId = str(uuid.uuid4())
        embedding = embeddings.embed_query(chunk["text"])
        metadata = {"page": "combined", "chunk": i, "source": chunk["metadata"]["source"]}
        
        all_data.append({
            "embeddingsId": embeddingsId,
            "text": chunk["text"],
            "embedding": embedding,
            "metadata": metadata
        })

    # ===== QDRANT OPERATIONS =====
    
    # Get a sample embedding to determine dimension
    embedding_dim = len(all_data[0]["embedding"])
    
    # First, try to delete the collection if it exists
    try:
        await qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception as e:
       print(f"Collection '{collection_name}' did not exist or could not be deleted: {e}")
       
    # Now, create the collection
    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": embedding_dim,
            "distance": "Cosine"
        }
    )
    print(f"Created collection '{collection_name}' with dimension {embedding_dim}.")

    points = [
        {
            "id": data["embeddingsId"],
            "vector": data["embedding"],
            "payload": {
                "text": data["text"],
                "metadata": data["metadata"]
            }
        } for data in all_data
    ]

    # Upserting points to Qdrant in batches
    qdrant_batch_size = 100  
    for i in range(0, len(points), qdrant_batch_size):
        batch = points[i:i+qdrant_batch_size]
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Upserted batch {i//qdrant_batch_size + 1} with {len(batch)} points to Qdrant")
    
    print(f"Completed all Qdrant operations: {len(points)} vectors inserted")

    # ===== NEO4J OPERATIONS =====
    # Clear Neo4j database first
    async with driver.session(database=neo4j_database) as session:
        await session.execute_write(clear_neo4j_database)
        print("Neo4j database cleared")

    async with driver.session(database=neo4j_database) as session:
            await session.execute_write(execute_cypher_queries)
            print("Neo4j schema rebuilt")
    
    # Prepare the list of points to upsert
    neo4j_batch_size = 50

    for i in range(0, len(all_data), neo4j_batch_size):
        batch_data = []
        # Create a new transaction
        for data in all_data[i:i+neo4j_batch_size]:
            concepts = [concept for concept in CONCEPTS_IN_GRAPH if concept.lower() in data["text"].lower()]
            batch_data.append({
                "embeddingsId": data["embeddingsId"],
                "text": data["text"],
                "metadata": json.dumps(data["metadata"]),
                "concepts": concepts
            })

        async with driver.session(database=neo4j_database) as session:
            await session.execute_write(create_chunk_and_concept_nodes, batch_data)
        print(f"Processed Neo4j batch {i//neo4j_batch_size + 1} with {len(batch_data)} chunks")

    print(f"Successfully processed {len(all_data)} chunks to Neo4j")
    print(f"Ingestion complete: {len(all_data)} chunks processed to both Qdrant and Neo4j")
    await driver.close()
    await qdrant_client.close()

async def execute_cypher_queries(tx):
    """Execute the cypher queries of nodes and relationships"""
    with open("knowledge_base.cypher", "r") as file:
        cypher_queries = file.read()

    # Execute the cypher queries
    await tx.run(cypher_queries)

async def create_chunk_node(tx, embeddingsId, text, metadata):
    """Create a chunk node in neo4j"""
    metadata_json = json.dumps(metadata)
    query = """
    CREATE (c:Chunk {embeddingsId: $embeddingsId, text: $text, metadata: $metadata})
    """
    await tx.run(query, embeddingsId=embeddingsId, text=text, metadata=metadata_json)

async def create_concept_node(tx, embeddingsId, chunk_text):
    for concept in CONCEPTS_IN_GRAPH:
        if concept.lower() in chunk_text.lower():
            query = """
            MATCH (c:Chunk {embeddingsId: $embeddingsId})
            MATCH (k:Concept {name: $concept})
            MERGE (c)-[:MENTIONS]->(k)
            """
            await tx.run(query, embeddingsId=embeddingsId, concept=concept)

async def create_chunk_and_concept_nodes(tx, batch_data):
    """Batch create chunk and concept nodes in Neo4j."""
    query = """
    UNWIND $data AS row
    CREATE (c:Chunk {embeddingsId: row.embeddingsId, text: row.text, metadata: row.metadata})
    WITH c, row
    UNWIND row.concepts AS concept
    MATCH (k:Concept {name: concept})
    MERGE (c)-[:MENTIONS]->(k)
    """
    await tx.run(query, data=batch_data)

async def clear_neo4j_database(session):
    """Clear all data from Neo4j database"""
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    await session.run(query)
    print("Neo4j database cleared successfully")

















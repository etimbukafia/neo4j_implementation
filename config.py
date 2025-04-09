import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def validate_environment():
    required_vars = ["QDRANT_URL", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return True

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD") 


def initialize_config():
    validate_environment()
    collection_name = "machine-learning"
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    neo4j_database = os.getenv("NEO4J_DATABASE")
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return qdrant_client, collection_name, neo4j_database, driver, embeddings






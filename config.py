# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration - Your specific setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Qz7wL_FBZ8yZnYLQskLTQKP1ajruNGdN4Cruje7L8jmA3Lo3i6bv8xAJ5dabpRqW87P3A")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "travel-assistant")
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://travel-assistant-k0vm7ai.svc.aped-4627-b74a.pinecone.io")

# Groq Configuration - Your specific setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Embedding Configuration - FIXED: Using compatible model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions - much better quality
EMBEDDING_DIM = 768  # Matches the model output

# Chat Configuration
MAX_COMPLETION_TOKENS = 1024
TEMPERATURE = 0.7
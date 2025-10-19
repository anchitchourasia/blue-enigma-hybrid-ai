# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud uses secrets.toml, not .env
def get_secret(key, default=None):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        # First try Streamlit secrets
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    # Fallback to environment variables
    return os.getenv(key, default)

# Import streamlit for secrets (only in deployed environment)
try:
    import streamlit as st
    USING_STREAMLIT = True
except:
    USING_STREAMLIT = False

# Configuration with fallbacks
def get_config_value(key, default=None):
    if USING_STREAMLIT and hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# Pinecone Configuration
PINECONE_API_KEY = get_config_value("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_config_value("PINECONE_INDEX_NAME", "travel-assistant")
PINECONE_HOST = get_config_value("PINECONE_HOST")

# Groq Configuration
GROQ_API_KEY = get_config_value("GROQ_API_KEY")
GROQ_MODEL = get_config_value("GROQ_MODEL", "llama-3.1-8b-instant")

# Neo4j Configuration (Optional)
NEO4J_URI = get_config_value("NEO4J_URI", "")
NEO4J_USERNAME = get_config_value("NEO4J_USERNAME", "")
NEO4J_PASSWORD = get_config_value("NEO4J_PASSWORD", "")

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
PINECONE_TARGET_DIM = 1024

# Chat Configuration
MAX_COMPLETION_TOKENS = 1024
TEMPERATURE = 0.7

# Check if we have minimum required APIs
def has_required_apis():
    return bool(PINECONE_API_KEY and GROQ_API_KEY)

# Demo mode flag
DEMO_MODE = not has_required_apis()

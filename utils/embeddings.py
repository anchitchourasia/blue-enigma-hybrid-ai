import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import groq
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import asyncio
from config import GROQ_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

class EmbeddingManager:
    def __init__(self):
        # Initialize Groq client for chat
        self.groq_client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Initialize local embedding model
        print(f"🔄 Loading local embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Local embedding model loaded successfully!")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local SentenceTransformer model"""
        try:
            if not texts:
                return []
            
            # Generate embeddings locally
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Your Pinecone expects 1024 dimensions, but our model gives 384
            # We'll pad the embeddings to match requirements
            padded_embeddings = []
            for embedding in embeddings:
                if len(embedding) < 1024:
                    # Pad with zeros to reach 1024 dimensions
                    padded = embedding + [0.0] * (1024 - len(embedding))
                    padded_embeddings.append(padded)
                else:
                    # Truncate if longer (unlikely)
                    padded_embeddings.append(embedding[:1024])
            
            return padded_embeddings
        except Exception as e:
            print(f"❌ Error generating local embeddings: {e}")
            # Return zero vectors as fallback (1024 dimensions)
            return [[0.0] * 1024 for _ in texts]
    
    def get_chat_completion(self, messages: List[dict], model: str = "llama-3.1-8b-instant") -> str:
        """Get chat completion using Groq with your specific parameters"""
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Error getting Groq completion: {e}")
            return f"I apologize, but I encountered an error with the AI service: {str(e)}"

    async def get_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_embeddings, texts
        )

# Global instance
embedding_manager = EmbeddingManager()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_manager.get_embeddings(texts)

def get_chat_completion(messages: List[dict], model: str = "llama-3.1-8b-instant") -> str:
    return embedding_manager.get_chat_completion(messages, model)

async def get_embeddings_async(texts: List[str]) -> List[List[float]]:
    return await embedding_manager.get_embeddings_async(texts)
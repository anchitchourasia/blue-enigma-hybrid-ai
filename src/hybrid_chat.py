import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone
from neo4j import GraphDatabase
import asyncio
from typing import List, Dict, Any, Tuple
import hashlib
import time
import re
from tqdm import tqdm

from config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    NEO4J_URI, 
    NEO4J_USERNAME, 
    NEO4J_PASSWORD
)
from utils.embeddings import get_embeddings, get_chat_completion

class HybridChatSystem:
    def __init__(self):
        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self.pinecone_index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
        
        # Initialize Neo4j client
        self.neo4j_driver = None
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            print("✅ Connected to Neo4j")
        except Exception as e:
            print(f"⚠️  Neo4j not available: {e}")
            print("💡 Continuing with Pinecone-only search")
            self.neo4j_driver = None
        
        # Cache for embeddings and results
        self.embedding_cache = {}
        
        # System prompt optimized for your dataset
        self.system_prompt = """You are VietnamTravel AI, an expert travel assistant specializing in Vietnam tourism. 

Use the provided context from travel databases to create personalized, practical itineraries and recommendations.

RESPONSE GUIDELINES:
1. Create detailed day-by-day itineraries when asked
2. Include specific cities, attractions, and practical tips
3. Consider best times to visit, regions, and travel connections
4. Suggest logical sequences and connections between locations
5. Be engaging but concise - focus on actionable advice
6. When referencing specific places, mention their names and key features
7. Provide 2-3 concrete itinerary steps or tips based on the context

Always base recommendations on the provided context data from the travel database."""

    def get_cached_embedding(self, text: str) -> List[float]:
        """Get cached embedding or compute new one using local model"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        embedding = get_embeddings([text])[0]
        self.embedding_cache[text_hash] = embedding
        return embedding

    async def query_pinecone_async(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query Pinecone asynchronously using local embeddings"""
        query_embedding = self.get_cached_embedding(query)
        
        try:
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            print(f"DEBUG: Pinecone found {len(results.get('matches', []))} results")
            return results.get('matches', [])
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            return []

    async def query_neo4j_async(self, query: str) -> List[Dict]:
        """Query Neo4j for contextual relationships - COMPLETELY FIXED VERSION"""
        if not self.neo4j_driver:
            return []
        
        key_terms = self.extract_key_terms(query)
        
        # COMPLETELY FIXED Neo4j query - only returns meaningful location data
        neo4j_query = """
        MATCH (loc:Location)
        WHERE any(term IN $terms WHERE 
               toLower(loc.name) CONTAINS toLower(term) OR
               toLower(loc.type) CONTAINS toLower(term) OR
               toLower(loc.region) CONTAINS toLower(term) OR
               toLower(loc.description) CONTAINS toLower(term))
        
        // Only get locations that have proper data
        AND loc.name IS NOT NULL 
        AND loc.name <> 'Unknown'
        AND loc.description IS NOT NULL
        AND size(loc.description) > 20
        
        OPTIONAL MATCH (loc)-[:LOCATED_IN]->(region:Region)
        OPTIONAL MATCH (loc)-[:HAS_TAG]->(tag:Tag)
        OPTIONAL MATCH (loc)-[:NEARBY]->(nearby:Location)
        
        WITH loc, region, 
             collect(DISTINCT tag.name) as tags, 
             collect(DISTINCT nearby.name) as nearby_locations
        
        RETURN 
            loc.id as node_id,
            loc.name as name,
            loc.type as type,
            loc.region as region,
            loc.description as description,
            loc.best_time_to_visit as best_time,
            tags,
            region.name as region_name,
            nearby_locations
        ORDER BY loc.name
        LIMIT 10
        """
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(neo4j_query, terms=key_terms)
                records = []
                for record in result:
                    record_data = {
                        "node_id": record.get("node_id", ""),
                        "name": record.get("name", ""),
                        "type": record.get("type", ""),
                        "region": record.get("region_name") or record.get("region", ""),
                        "description": record.get("description", ""),
                        "best_time": record.get("best_time", ""),
                        "tags": record.get("tags", []),
                        "nearby_locations": record.get("nearby_locations", [])
                    }
                    records.append(record_data)
                
                print(f"DEBUG: Neo4j found {len(records)} meaningful locations")
                return records
        except Exception as e:
            print(f"❌ Error querying Neo4j: {e}")
            return []

    def extract_key_terms(self, query: str) -> List[str]:
        """Enhanced key term extraction for Vietnam travel"""
        query_lower = query.lower()
        
        # Vietnam locations from your dataset
        vietnam_locations = [
            'hanoi', 'ha long', 'halong', 'hue', 'hoi an', 'da nang', 'nha trang',
            'saigon', 'ho chi minh', 'mekong', 'sapa', 'ninh binh', 'phu quoc',
            'northern', 'central', 'southern', 'da lat'
        ]
        
        # Travel-related terms
        travel_terms = [
            'romantic', 'adventure', 'budget', 'luxury', 'family', 'solo', 'couple',
            'beach', 'mountain', 'culture', 'food', 'historical', 'itinerary', 'tour',
            'relax', 'explore', 'hiking', 'cruise', 'island', 'city', 'countryside'
        ]
        
        extracted = []
        extracted.extend([term for term in travel_terms if term in query_lower])
        extracted.extend([loc for loc in vietnam_locations if loc in query_lower])
        
        # Extract number of days
        day_matches = re.findall(r'(\d+)\s*day', query_lower)
        extracted.extend(day_matches)
        
        return extracted if extracted else ['vietnam', 'travel']

    async def hybrid_search(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """Perform hybrid search using both Pinecone and Neo4j"""
        # Run both queries concurrently
        pinecone_task = self.query_pinecone_async(query)
        neo4j_task = self.query_neo4j_async(query) if self.neo4j_driver else asyncio.sleep(0)
        
        pinecone_results = await pinecone_task
        neo4j_results = await neo4j_task if self.neo4j_driver else []
        
        return pinecone_results, neo4j_results

    async def hybrid_search_with_metrics(self, query: str) -> Tuple[List[Dict], List[Dict], float]:
        """Perform hybrid search and return results with timing"""
        start_time = time.time()
        pinecone_results, neo4j_results = await self.hybrid_search(query)
        search_time = time.time() - start_time
        
        return pinecone_results, neo4j_results, search_time

    def build_prompt(self, user_query: str, pinecone_matches: List[Dict], graph_facts: List[Dict]) -> List[Dict]:
        """Build chat prompt combining vector DB matches and graph facts"""
        
        # Build vector context from Pinecone results
        vec_context = []
        for i, match in enumerate(pinecone_matches[:5], 1):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)
            
            snippet = f"{i}. {metadata.get('name', 'Unknown')} "
            snippet += f"(Type: {metadata.get('type', 'Unknown')}, "
            snippet += f"Region: {metadata.get('region', 'Unknown')}) - "
            snippet += f"{metadata.get('description', metadata.get('semantic_text', ''))} "
            
            tags = metadata.get('tags', [])
            if tags:
                snippet += f"Tags: {', '.join(tags)}. "
                
            best_time = metadata.get('best_time_to_visit', 'Not specified')
            snippet += f"Best time: {best_time}. "
            snippet += f"Relevance score: {score:.3f}"
            
            vec_context.append(snippet)

        # Build graph context from Neo4j results - IMPROVED
        graph_context = []
        
        for fact in graph_facts[:8]:
            snippet = f"• {fact.get('name', 'Unknown')} "
            snippet += f"({fact.get('type', 'Unknown')}) "
            snippet += f"in {fact.get('region', 'Unknown')}: "
            snippet += f"{fact.get('description', '')} "
            
            # Add tags if available
            tags = fact.get('tags', [])
            if tags:
                snippet += f"Features: {', '.join(tags[:3])}. "
                
            # Add nearby locations if available
            nearby = fact.get('nearby_locations', [])
            if nearby:
                snippet += f"Nearby destinations: {', '.join(nearby[:2])}. "
                
            graph_context.append(snippet)

        # Build final prompt
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": 
             f"User query: {user_query}\n\n"
             "Top travel destinations from database:\n" + "\n".join(vec_context) + "\n\n"
             "Location connections and context:\n" + "\n".join(graph_context) + "\n\n"
             "Based on the above Vietnam travel data, provide a helpful response. Include specific cities, attractions, and practical travel advice."}
        ]
        
        return prompt

    def generate_response(self, query: str, pinecone_results: List[Dict], neo4j_results: List[Dict]) -> str:
        """Generate response using the enhanced prompt building"""
        prompt = self.build_prompt(query, pinecone_results, neo4j_results)
        
        try:
            response = get_chat_completion(prompt, GROQ_MODEL)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"

    async def process_query_with_metrics(self, query: str) -> Tuple[List[Dict], List[Dict], str, float]:
        """Process query and return results with full metrics"""
        search_start = time.time()
        pinecone_results, neo4j_results, search_time = await self.hybrid_search_with_metrics(query)
        
        # Generate response
        response_start = time.time()
        response = self.generate_response(query, pinecone_results, neo4j_results)
        response_time = time.time() - response_start
        
        total_time = time.time() - search_start
        
        print(f"✅ Search: {search_time:.2f}s, Response: {response_time:.2f}s, Total: {total_time:.2f}s")
        
        return pinecone_results, neo4j_results, response, total_time

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring"""
        status = {
            "pinecone_connected": True,
            "neo4j_connected": self.neo4j_driver is not None,
            "groq_configured": bool(GROQ_API_KEY),
            "embedding_model_loaded": True
        }
        
        # Test Neo4j connection if available
        if self.neo4j_driver:
            try:
                self.neo4j_driver.verify_connectivity()
                status["neo4j_connected"] = True
            except:
                status["neo4j_connected"] = False
        
        return status

    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

# Interactive chat interface
def interactive_chat():
    """Interactive chat interface"""
    print("🤖 Vietnam Travel Assistant - Hybrid AI System")
    print(f"🚀 Powered by Groq ({GROQ_MODEL}) & Pinecone")
    print("Type 'exit' to quit.\n")
    
    chat_system = HybridChatSystem()
    
    try:
        while True:
            query = input("\nEnter your travel question: ").strip()
            if not query or query.lower() in ("exit", "quit"):
                break

            print("Processing your query...")
            start_time = time.time()
            
            try:
                pinecone_results, neo4j_results, response, total_time = asyncio.run(
                    chat_system.process_query_with_metrics(query)
                )
                
                print(f"\n=== Assistant Answer ===\n")
                print(response)
                print(f"\nSearch completed in {total_time:.2f}s")
                print(f"Found {len(pinecone_results)} vector results and {len(neo4j_results)} graph results")
                print("\n=== End ===\n")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        chat_system.close()

if __name__ == "__main__":
    interactive_chat()
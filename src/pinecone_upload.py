import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from tqdm import tqdm
from typing import List, Dict, Any
import re

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from utils.embeddings import get_embeddings

class PineconeUploader:
    def __init__(self):
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
    
    def load_dataset(self, filepath: str = "data/vietnam_travel_dataset.txt") -> List[Dict[str, Any]]:
        """Load dataset with fixed-width like parsing for space-separated format"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            if not lines:
                return []
            
            # Parse header
            header_line = lines[0]
            print(f"🔍 Raw header: {header_line[:100]}...")
            
            # The headers are space-separated, but we need to be careful with multi-word headers
            headers = [
                'id', 'type', 'name', 'region', 'description', 
                'best_time_to_visit', 'tags/0', 'tags/1', 'tags/2', 
                'semantic_text', 'connections/0/relation', 'connections/0/target', 
                'connections/1/relation', 'connections/1/target', 'city'
            ]
            
            print(f"📋 Using predefined {len(headers)} headers")
            
            data = []
            successful_lines = 0
            
            for line_num, line in enumerate(lines[1:], 2):  # Skip header
                line = line.strip()
                if not line:
                    continue
                
                # Use a smarter parsing approach for space-separated data with long descriptions
                item = self.parse_fixed_width_line(line, headers)
                if item and item.get('id'):
                    data.append(item)
                    successful_lines += 1
                else:
                    print(f"⚠️  Line {line_num}: Could not parse - {line[:50]}...")
            
            print(f"✅ Successfully parsed {successful_lines} items out of {len(lines)-1} lines")
            return data
            
        except Exception as e:
            print(f"❌ Error parsing dataset: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def parse_fixed_width_line(self, line: str, headers: List[str]) -> Dict[str, Any]:
        """Parse a line with fixed-width like parsing"""
        try:
            # Split by 2 or more spaces to handle the format
            parts = re.split(r'\s{2,}', line)
            
            if len(parts) < 5:  # Need at least id, type, name, region, description
                return {}
            
            item = {}
            
            # Map parts to headers
            for i, header in enumerate(headers):
                if i < len(parts) and parts[i].strip():
                    item[header] = parts[i].strip()
                else:
                    item[header] = ""
            
            return item
            
        except Exception as e:
            print(f"   Parse error: {e}")
            return {}
    
    def create_semantic_text(self, item: Dict[str, Any]) -> str:
        """Create rich semantic text from item data"""
        parts = []
        
        # Extract basic info
        name = item.get('name', item.get('id', 'Unknown')).replace('_', ' ').title()
        item_type = item.get('type', 'Location')
        region = item.get('region', 'Vietnam')
        
        parts.append(f"{name} is a {item_type.lower()} in {region}.")
        
        # Add description
        description = item.get('description', '')
        if description:
            parts.append(f"{description}")
        
        # Add best time to visit
        best_time = item.get('best_time_to_visit', '')
        if best_time:
            parts.append(f"Best time to visit: {best_time}.")
        
        # Add tags
        tags = []
        for i in range(3):  # tags/0, tags/1, tags/2
            tag_key = f'tags/{i}'
            if tag_key in item and item[tag_key]:
                tags.append(item[tag_key])
        
        if tags:
            tags_text = ", ".join(tags)
            parts.append(f"Features: {tags_text}.")
        
        # Add semantic text
        semantic_text = item.get('semantic_text', '')
        if semantic_text:
            parts.append(f"Overview: {semantic_text}")
        
        return " ".join(parts)
    
    def prepare_vectors(self, data: List[Dict[str, Any]]) -> List[Dict]:
        """Prepare vectors with proper dimension handling"""
        vectors = []
        
        for item in tqdm(data, desc="🔄 Creating vectors"):
            try:
                # Create semantic text
                semantic_text = self.create_semantic_text(item)
                
                if not semantic_text or len(semantic_text) < 10:
                    print(f"⚠️  Skipping item with insufficient text: {item.get('id', 'unknown')}")
                    continue
                
                # Generate embedding
                embedding = get_embeddings([semantic_text])[0]
                
                # Ensure embedding is 1024 dimensions
                if len(embedding) != 1024:
                    print(f"⚠️  Embedding dimension is {len(embedding)}, padding to 1024")
                    if len(embedding) < 1024:
                        embedding = embedding + [0.0] * (1024 - len(embedding))
                    else:
                        embedding = embedding[:1024]
                
                # Create vector ID
                item_id = str(item.get('id', f"item_{hash(semantic_text)}"))
                
                # Extract tags
                tags = []
                for i in range(3):
                    tag_key = f'tags/{i}'
                    if tag_key in item and item[tag_key]:
                        tags.append(item[tag_key])
                
                # Create vector
                vector = {
                    "id": item_id,
                    "values": embedding,
                    "metadata": {
                        "name": item.get('name', 'Unknown').replace('_', ' ').title(),
                        "type": item.get('type', 'Unknown'),
                        "region": item.get('region', 'Unknown'),
                        "description": item.get('description', ''),
                        "semantic_text": semantic_text,
                        "best_time_to_visit": item.get('best_time_to_visit', ''),
                        "tags": tags,
                    }
                }
                vectors.append(vector)
                
            except Exception as e:
                print(f"❌ Error processing item {item.get('id', 'unknown')}: {e}")
                continue
        
        return vectors
    
    def upload_vectors(self, vectors: List[Dict], batch_size: int = 50):
        """Upload vectors to Pinecone"""
        if not vectors:
            print("❌ No vectors to upload!")
            return 0
            
        print(f"📤 Uploading {len(vectors)} vectors in batches of {batch_size}")
        
        successful_uploads = 0
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
            batch = vectors[i:i + batch_size]
            try:
                # Verify first vector dimension
                if batch and len(batch[0]['values']) != 1024:
                    print(f"❌ Vector dimension mismatch: {len(batch[0]['values'])}")
                    continue
                    
                self.index.upsert(vectors=batch)
                successful_uploads += len(batch)
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ Error uploading batch {i//batch_size}: {e}")
        
        return successful_uploads
    
    def get_index_stats(self):
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            print(f"📊 Index Statistics:")
            print(f"   - Total Vectors: {stats.get('total_vector_count', 0)}")
            print(f"   - Dimension: {stats.get('dimension', 0)}")
            return stats
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}

def main():
    """Main function to run Pinecone upload"""
    print("🚀 Starting Pinecone Upload Process")
    print(f"📝 Target Index: {PINECONE_INDEX_NAME}")
    
    try:
        # Initialize uploader
        uploader = PineconeUploader()
        
        # Load and parse dataset
        data = uploader.load_dataset()
        if not data:
            print("❌ No data loaded. Exiting.")
            return
        
        print(f"📊 Loaded {len(data)} items")
        
        # Show sample for debugging
        if data:
            sample = data[0]
            print(f"📄 Sample item:")
            print(f"   ID: {sample.get('id', 'Not found')}")
            print(f"   Name: {sample.get('name', 'Not found')}")
            print(f"   Type: {sample.get('type', 'Not found')}")
            print(f"   Region: {sample.get('region', 'Not found')}")
            print(f"   Description: {sample.get('description', '')[:100]}...")
        
        # Prepare vectors
        vectors = uploader.prepare_vectors(data)
        print(f"✅ Prepared {len(vectors)} vectors")
        
        if vectors:
            print(f"🔢 First vector dimension: {len(vectors[0]['values'])}")
        
        # Upload vectors
        successful_uploads = uploader.upload_vectors(vectors, batch_size=20)
        
        if successful_uploads > 0:
            uploader.get_index_stats()
            print(f"🎉 Successfully uploaded {successful_uploads} vectors!")
        else:
            print("❌ No vectors were uploaded.")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
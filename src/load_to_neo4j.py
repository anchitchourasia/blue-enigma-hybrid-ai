import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
import json
from typing import List, Dict, Any
from tqdm import tqdm
import re
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class Neo4jLoader:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j")
        except Exception as e:
            print(f"❌ Could not connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def is_connected(self):
        return self.driver is not None
    
    def clear_database(self):
        """Clear existing data"""
        if not self.is_connected():
            return
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared")
    
    def create_constraints(self):
        """Create database constraints"""
        if not self.is_connected():
            return
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"⚠️  Could not create constraint: {e}")
    
    def load_text_dataset(self, filepath: str = "data/vietnam_travel_dataset.txt") -> List[Dict[str, Any]]:
        """Load and clean travel dataset"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            if not lines:
                return []
            
            data = []
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if not line:
                    continue
                
                # Improved parsing with better field handling
                parts = re.split(r'\s{2,}', line)
                
                if len(parts) >= 5:
                    item = {
                        'id': parts[0] if len(parts) > 0 else '',
                        'type': parts[1] if len(parts) > 1 else 'Location',
                        'name': parts[2] if len(parts) > 2 else '',
                        'region': parts[3] if len(parts) > 3 else 'Vietnam',
                        'description': parts[4] if len(parts) > 4 else '',
                        'best_time_to_visit': parts[5] if len(parts) > 5 else 'Year-round',
                        'tags': []
                    }
                    
                    # Extract tags (positions 6-8)
                    for i in range(6, 9):
                        if i < len(parts) and parts[i].strip():
                            item['tags'].append(parts[i].strip())
                    
                    # Only include items with valid names
                    if item['name'] and item['name'] != 'Unknown':
                        data.append(item)
            
            print(f"✅ Loaded {len(data)} valid travel items")
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
    
    def load_data(self):
        """Load travel data into Neo4j with meaningful relationships"""
        if not self.is_connected():
            print("⚠️  Skipping data load - no Neo4j connection")
            return
        
        data = self.load_text_dataset()
        if not data:
            print("❌ No data to load")
            return
        
        print(f"📥 Loading {len(data)} travel destinations into Neo4j...")
        
        with self.driver.session() as session:
            for item in tqdm(data, desc="Creating nodes"):
                try:
                    # Skip items with missing critical data
                    if not item.get('name') or not item.get('id'):
                        continue
                    
                    # Create Location node with all properties
                    session.run("""
                        MERGE (l:Location {id: $id})
                        SET l.name = $name,
                            l.type = $type,
                            l.region = $region,
                            l.description = $description,
                            l.best_time_to_visit = $best_time
                    """, 
                    id=item['id'],
                    name=item['name'],
                    type=item.get('type', 'Location'),
                    region=item.get('region', 'Vietnam'),
                    description=item.get('description', ''),
                    best_time=item.get('best_time_to_visit', 'Year-round'))
                    
                    # Create Region connection
                    region = item.get('region', 'Vietnam')
                    if region:
                        session.run("""
                            MERGE (r:Region {name: $region})
                            MERGE (l:Location {id: $id})
                            MERGE (l)-[:LOCATED_IN]->(r)
                        """, region=region, id=item['id'])
                    
                    # Create Tag connections
                    tags = item.get('tags', [])
                    for tag in tags:
                        if tag and tag.strip():
                            session.run("""
                                MERGE (t:Tag {name: $tag})
                                MERGE (l:Location {id: $id})
                                MERGE (l)-[:HAS_TAG]->(t)
                            """, tag=tag.strip(), id=item['id'])
                    
                    # Create meaningful connections between locations in same region
                    if item.get('region'):
                        session.run("""
                            MATCH (current:Location {id: $id})
                            MATCH (other:Location {region: $region})
                            WHERE current.id <> other.id
                            MERGE (current)-[:NEARBY]->(other)
                        """, id=item['id'], region=item['region'])
                            
                except Exception as e:
                    print(f"❌ Error processing item {item.get('id')}: {e}")
                    continue
        
        print("✅ Neo4j data loading completed!")

def main():
    """Main function to load data into Neo4j"""
    print("🚀 Starting Neo4j Data Loading...")
    
    loader = Neo4jLoader()
    
    try:
        if loader.is_connected():
            # Clear existing data
            loader.clear_database()
            
            # Create constraints first
            loader.create_constraints()
            
            # Load data
            loader.load_data()
            
            print("🎉 Neo4j loading completed successfully!")
        else:
            print("⚠️  Neo4j not available - continuing without graph database")
        
    except Exception as e:
        print(f"❌ Error during Neo4j loading: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()
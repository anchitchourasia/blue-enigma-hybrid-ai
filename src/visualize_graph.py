import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class GraphVisualizer:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def get_graph_data(self):
        """Extract graph data from Neo4j"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN 
            labels(n)[0] as source_label,
            n.name as source_name,
            type(r) as relationship,
            labels(m)[0] as target_label,
            m.name as target_name
        LIMIT 200
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def create_network_graph(self):
        """Create an interactive network graph using PyVis"""
        data = self.get_graph_data()
        
        # Create network
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add nodes and edges
        nodes = set()
        for record in data:
            source = f"{record['source_label']}: {record['source_name']}"
            target = f"{record['target_label']}: {record['target_name']}"
            
            if source not in nodes:
                net.add_node(source, label=record['source_name'], 
                           title=record['source_label'], group=record['source_label'])
                nodes.add(source)
            
            if target not in nodes:
                net.add_node(target, label=record['target_name'], 
                           title=record['target_label'], group=record['target_label'])
                nodes.add(target)
            
            net.add_edge(source, target, title=record['relationship'])
        
        return net
    
    def show_basic_stats(self):
        """Display basic graph statistics"""
        queries = {
            "Total Nodes": "MATCH (n) RETURN count(n) as count",
            "Total Relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "Locations": "MATCH (l:Location) RETURN count(l) as count",
            "Activities": "MATCH (a:Activity) RETURN count(a) as count",
            "Places": "MATCH (p:Place) RETURN count(p) as count"
        }
        
        stats = {}
        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                stats[name] = result.single()["count"]
        
        return stats

def main():
    """Main function to visualize the graph"""
    print("🚀 Starting Graph Visualization...")
    
    visualizer = GraphVisualizer()
    
    try:
        # Show statistics
        stats = visualizer.show_basic_stats()
        print("📊 Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Create and save interactive visualization
        net = visualizer.create_network_graph()
        net.show("travel_graph.html")
        print("✅ Interactive graph saved as 'travel_graph.html'")
        
        print("🎉 Graph visualization completed!")
        
    finally:
        visualizer.close()

if __name__ == "__main__":
    main()
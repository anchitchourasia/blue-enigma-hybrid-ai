import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

def debug_pinecone():
    """Debug Pinecone connection and index status"""
    print("🔧 Pinecone Debug Information")
    print("=" * 50)
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"✅ Pinecone client initialized")
        
        # List all indexes
        indexes = pc.list_indexes()
        print(f"📋 Available indexes: {[idx.name for idx in indexes.indexes]}")
        
        # Connect to specific index
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to index: {PINECONE_INDEX_NAME}")
        
        # Get detailed stats
        stats = index.describe_index_stats()
        print(f"📊 Index Statistics:")
        print(f"   - Total Vectors: {stats.get('total_vector_count', 0)}")
        print(f"   - Dimension: {stats.get('dimension', 0)}")
        
        # Show all namespaces
        namespaces = stats.get('namespaces', {})
        if namespaces:
            print("   - Namespaces:")
            for ns, details in namespaces.items():
                print(f"     '{ns}': {details.get('vector_count', 0)} vectors")
        else:
            print("   - No namespaces found (using default)")
            
        # Test vector operations
        print("\n🧪 Testing vector operations...")
        
        # Try to upsert a test vector
        test_vector = {
            "id": "test_vector_123",
            "values": [0.1] * 1024,  # 1024-dimensional vector
            "metadata": {"test": True, "name": "Test Vector"}
        }
        
        # Try different namespaces
        namespaces_to_test = ["", "default", None]
        
        for ns in namespaces_to_test:
            try:
                if ns == "":
                    print(f"🔍 Testing default namespace (empty string)...")
                    index.upsert(vectors=[test_vector], namespace="")
                elif ns is None:
                    print(f"🔍 Testing None namespace...")
                    index.upsert(vectors=[test_vector])
                else:
                    print(f"🔍 Testing namespace '{ns}'...")
                    index.upsert(vectors=[test_vector], namespace=ns)
                
                print(f"   ✅ Successfully upserted test vector to namespace: {ns}")
                
                # Check if vector exists
                time.sleep(1)  # Wait for upsert to propagate
                fetch_result = index.fetch(ids=["test_vector_123"], namespace=ns if ns != "" else None)
                if fetch_result and 'test_vector_123' in fetch_result.vectors:
                    print(f"   ✅ Vector found in namespace: {ns}")
                else:
                    print(f"   ❌ Vector not found in namespace: {ns}")
                    
                # Clean up
                index.delete(ids=["test_vector_123"], namespace=ns if ns != "" else None)
                
            except Exception as e:
                print(f"   ❌ Error with namespace '{ns}': {e}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pinecone()
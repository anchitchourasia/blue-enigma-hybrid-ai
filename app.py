# app.py
import streamlit as st
import asyncio
import time
import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configure page first
st.set_page_config(
    page_title="Vietnam Travel AI Assistant",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed CSS with proper text contrast
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .response-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .response-text {
        color: #2c3e50 !important;
        font-size: 16px;
        line-height: 1.7;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .response-text h1, .response-text h2, .response-text h3, .response-text h4 {
        color: #1f77b4 !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    .response-text strong {
        color: #1f77b4 !important;
        font-weight: 700;
    }
    .response-text ul, .response-text ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    .response-text li {
        margin-bottom: 0.5rem;
        color: #2c3e50 !important;
    }
    .response-text p {
        margin-bottom: 1rem;
        color: #2c3e50 !important;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-box h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .quick-query-btn {
        background: #e9ecef !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
        margin: 0.25rem 0;
    }
    .quick-query-btn:hover {
        background: #dee2e6 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Ensure all text in main content is visible */
    .main .block-container {
        color: #333333;
    }
    
    /* Fix expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* Custom card style for search results */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def safe_import():
    """Safely import required modules with error handling"""
    try:
        from src.hybrid_chat import HybridChatSystem
        from utils.embeddings import get_embeddings
        return HybridChatSystem, get_embeddings, None
    except ImportError as e:
        return None, None, f"Import error: {e}"
    except Exception as e:
        return None, None, f"Initialization error: {e}"

class StreamlitTravelApp:
    def __init__(self):
        self.chat_system = None
        self.import_error = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the hybrid chat system with error handling"""
        try:
            with st.spinner("🚀 Initializing AI Travel Assistant..."):
                HybridChatSystemClass, _, error = safe_import()
                if error:
                    self.import_error = error
                    return
                
                self.chat_system = HybridChatSystemClass()
                st.success("✅ AI System Ready!")
        except Exception as e:
            self.import_error = f"Failed to initialize: {str(e)}"
            st.error(f"❌ {self.import_error}")
    
    def display_welcome_section(self):
        """Display welcome section with app information"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 class="main-header">🌏 Vietnam Travel AI Assistant</h1>
            <p style='font-size: 1.2rem; color: #666; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
            Your intelligent travel companion powered by Hybrid AI - combining Pinecone vector search, 
            Neo4j knowledge graphs, and Groq's lightning-fast LLM for personalized Vietnam travel recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_sidebar(self):
        """Display sidebar with information and quick queries"""
        with st.sidebar:
            st.markdown("### ⚡ Quick Queries")
            st.markdown("*Click any query below to get started:*")
            
            quick_queries = [
                "Create a romantic 4 day itinerary for Vietnam",
                "Best beach destinations in Vietnam",
                "Adventure activities in northern Vietnam", 
                "Cultural and historical sites in central Vietnam",
                "Budget travel options in Vietnam",
                "Luxury honeymoon destinations in Vietnam",
                "Family-friendly activities in Vietnam",
                "Best food experiences in Vietnam"
            ]
            
            for query in quick_queries:
                if st.button(f"🗨️ {query}", key=query, use_container_width=True):
                    st.session_state.user_input = query
                    if 'process_query' in st.session_state:
                        st.session_state.process_query = True
            
            st.markdown("---")
            st.markdown("### 🔧 System Status")
            
            if self.import_error:
                st.error("❌ System Error")
                st.write(f"*Details:* {self.import_error}")
            elif self.chat_system:
                try:
                    status = self.chat_system.get_system_status()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pinecone", "✅" if status["pinecone_connected"] else "❌")
                        st.metric("Groq", "✅" if status["groq_configured"] else "❌")
                    with col2:
                        st.metric("Neo4j", "✅" if status["neo4j_connected"] else "⚠️")
                        st.metric("Embeddings", "✅" if status["embedding_model_loaded"] else "❌")
                except Exception as e:
                    st.info("⚠️ Status check unavailable")
                    st.write(f"*Error:* {str(e)}")
            else:
                st.warning("🔧 System initializing...")
            
            st.markdown("---")
            st.markdown("""
            **💡 Powered by:**
            - 🗃️ **Pinecone** - Vector Search
            - 🕸️ **Neo4j** - Knowledge Graph  
            - 🤖 **Groq LLM** - AI Reasoning
            - 🔤 **Local Embeddings** - No API costs
            
            **🎯 Features:**
            - Hybrid AI search
            - Personalized itineraries
            - Real-time recommendations
            - Budget planning
            """)
    
    def display_search_metrics(self, metrics):
        """Display search performance metrics"""
        if not metrics:
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>🔍 Vector Results</h3>
                <h2>{metrics.get('vector_results', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>🕸️ Location Results</h3>
                <h2>{metrics.get('graph_results', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>⏱️ Search Time</h3>
                <h2>{metrics.get('search_time', 0):.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <h3>⚡ Total Time</h3>
                <h2>{metrics.get('total_time', 0):.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
    
    def display_search_results(self, pinecone_results, neo4j_results):
        """Display detailed search results with improved graph data"""
        if not pinecone_results and not neo4j_results:
            st.info("🔍 No search results found. Try a different query.")
            return
            
        st.markdown("### 📊 Search Results")
        
        # Pinecone Results
        if pinecone_results:
            with st.expander(f"🗃️ Vector Search Results ({len(pinecone_results)} found)", expanded=True):
                for i, result in enumerate(pinecone_results[:5], 1):
                    metadata = result.get('metadata', {})
                    score = result.get('score', 0)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: between; align-items: start;">
                            <div style="flex: 1;">
                                <h4 style="margin: 0; color: #1f77b4;">{i}. {metadata.get('name', 'Unknown')}</h4>
                                <p style="margin: 0.25rem 0; color: #666;">
                                    📍 <strong>Location:</strong> {metadata.get('region', 'Unknown')} | 
                                    🏷️ <strong>Type:</strong> {metadata.get('type', 'Unknown')}
                                </p>
                                <p style="margin: 0.5rem 0; color: #333;">{metadata.get('description', 'No description')}</p>
                                <p style="margin: 0.25rem 0; color: #555;">
                                    🎯 <strong>Tags:</strong> {', '.join(metadata.get('tags', []))}
                                </p>
                                <p style="margin: 0.25rem 0; color: #555;">
                                    📅 <strong>Best Time:</strong> {metadata.get('best_time_to_visit', 'Not specified')}
                                </p>
                            </div>
                            <div style="margin-left: 1rem;">
                                <div style="background: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 20px; text-align: center;">
                                    <strong>{score:.3f}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # COMPLETELY FIXED Neo4j Results Display
        if neo4j_results:
            # Filter out any remaining poor quality results
            clean_neo4j_results = [
                r for r in neo4j_results 
                if r.get('name') and r.get('name') != 'Unknown' 
                and r.get('description') and len(r.get('description', '')) > 20
            ]
            
            with st.expander(f"🗺️ Related Locations ({len(clean_neo4j_results)} found)"):
                if not clean_neo4j_results:
                    st.info("No meaningful location relationships found for this query.")
                    return
                    
                for i, result in enumerate(clean_neo4j_results[:6], 1):
                    tags = result.get('tags', [])
                    nearby = result.get('nearby_locations', [])
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4 style="margin: 0; color: #1f77b4;">{i}. {result.get('name', 'Unknown')}</h4>
                        <p style="margin: 0.25rem 0; color: #666;">
                            📍 <strong>Region:</strong> {result.get('region', 'Unknown')} | 
                            🏷️ <strong>Type:</strong> {result.get('type', 'Unknown')}
                        </p>
                        <p style="margin: 0.5rem 0; color: #333;">{result.get('description', 'No description')}</p>
                        <p style="margin: 0.25rem 0; color: #555;">
                            📅 <strong>Best Time to Visit:</strong> {result.get('best_time', 'Not specified')}
                        </p>
                        {f'<p style="margin: 0.25rem 0; color: #888;">🏷️ <strong>Features:</strong> {", ".join(tags[:3])}</p>' if tags else ''}
                        {f'<p style="margin: 0.25rem 0; color: #888;">🔗 <strong>Nearby Destinations:</strong> {", ".join(nearby[:2])}</p>' if nearby else ''}
                    </div>
                    """, unsafe_allow_html=True)
    
    def display_response(self, response):
        """Display the AI response with perfect visibility"""
        st.markdown("### 🧠 Travel Assistant Response")
        
        # Enhanced response display with guaranteed visibility
        st.markdown(
            f"""
            <div class="response-container">
                <div class="response-text">
                    {response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def create_itinerary_visualization(self, response):
        """Create a simple visualization for itineraries"""
        if not response or ("itinerary" not in response.lower() and "day" not in response.lower()):
            return
            
        st.markdown("### 📅 Itinerary Overview")
        
        # Simple extraction of days from response
        lines = response.split('\n')
        days = []
        current_day = {"title": "", "activities": []}
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Detect day headers
            if any(day_indicator in line_lower for day_indicator in ['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'itinerary']):
                if current_day["title"]:  # Save previous day if exists
                    days.append(current_day)
                current_day = {"title": line, "activities": []}
            # Detect activities (lines with time or action words)
            elif current_day["title"] and line and len(line) > 10:
                if any(indicator in line_lower for indicator in ['am', 'pm', 'morning', 'afternoon', 'evening', 'breakfast', 'lunch', 'dinner', 'visit', 'explore', 'enjoy']):
                    current_day["activities"].append(line)
        
        # Add the last day
        if current_day["title"]:
            days.append(current_day)
        
        if days:
            for day in days:
                with st.expander(f"📌 {day['title']}", expanded=True):
                    if day['activities']:
                        for activity in day['activities'][:6]:  # Show first 6 activities
                            st.write(f"• {activity}")
                    else:
                        # Fallback: show some context from the response
                        st.info("Detailed activities available in the main response above")
        else:
            # Fallback visualization
            st.info("✨ **Trip Highlights**")
            notable_lines = [line.strip() for line in lines if len(line.strip()) > 30 and not line.strip().startswith('**')]
            for line in notable_lines[:6]:
                st.write(f"• {line}")
    
    async def process_user_query(self, query):
        """Process user query and return results"""
        if not self.chat_system:
            return None, None, None, None, "System not initialized"
        
        try:
            start_time = time.time()
            
            # Use the new method with metrics
            pinecone_results, neo4j_results, response, total_time = await self.chat_system.process_query_with_metrics(query)
            
            search_time = total_time * 0.6  # Estimate search time
            
            metrics = {
                'vector_results': len(pinecone_results),
                'graph_results': len(neo4j_results),
                'search_time': search_time,
                'total_time': total_time
            }
            
            return pinecone_results, neo4j_results, response, metrics, None
            
        except Exception as e:
            return None, None, None, None, f"Error processing query: {str(e)}"
    
    def run(self):
        """Main method to run the Streamlit app"""
        # Initialize session state
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        if 'last_response' not in st.session_state:
            st.session_state.last_response = None
        if 'process_query' not in st.session_state:
            st.session_state.process_query = False
        
        self.display_welcome_section()
        self.display_sidebar()
        
        # Main input area
        st.markdown("### 💬 Ask Your Travel Question")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Enter your travel question:",
                value=st.session_state.user_input,
                placeholder="e.g., Create a romantic 4 day itinerary for Vietnam...",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.button("🚀 Generate", use_container_width=True)
        
        # Process query if submitted
        if (submit_button or st.session_state.process_query) and user_input:
            st.session_state.process_query = False
            st.session_state.user_input = user_input
            
            if not self.chat_system:
                st.error("❌ AI system not available. Please check the system status in the sidebar.")
                return
            
            # Create a placeholder for results
            results_placeholder = st.empty()
            
            with results_placeholder.container():
                with st.spinner("🔍 Searching travel database and generating response..."):
                    # Process the query
                    try:
                        # Run async function
                        pinecone_results, neo4j_results, response, metrics, error = asyncio.run(
                            self.process_user_query(user_input)
                        )
                        
                        if error:
                            st.error(f"❌ {error}")
                            return
                        
                        # Store in session state
                        st.session_state.last_response = {
                            'query': user_input,
                            'response': response,
                            'pinecone_results': pinecone_results,
                            'neo4j_results': neo4j_results,
                            'metrics': metrics
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
        
        # Display results if available
        if st.session_state.last_response:
            data = st.session_state.last_response
            
            # Display metrics
            self.display_search_metrics(data['metrics'])
            
            # Display search results
            self.display_search_results(data['pinecone_results'], data['neo4j_results'])
            
            # Display AI response
            self.display_response(data['response'])
            
            # Create visualization for itineraries
            self.create_itinerary_visualization(data['response'])

def main():
    # Initialize and run the app
    app = StreamlitTravelApp()
    app.run()

if __name__ == "__main__":
    main()
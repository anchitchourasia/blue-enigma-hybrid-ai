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
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
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
    """Safely import required modules with comprehensive error handling"""
    try:
        # Check if we're in Streamlit Cloud
        is_streamlit_cloud = 'STREAMLIT_DEPLOYMENT' in os.environ
        
        if is_streamlit_cloud:
            st.info("🌐 Running in Streamlit Cloud environment")
        
        # Try to import main components
        from src.hybrid_chat import HybridChatSystem
        from utils.embeddings import get_embeddings
        
        # Check if config is properly set up
        from config import PINECONE_API_KEY, GROQ_API_KEY, has_required_apis
        
        if not has_required_apis():
            return None, None, "Missing API keys. Please check your Streamlit secrets configuration."
        
        return HybridChatSystem, get_embeddings, None
        
    except ImportError as e:
        error_msg = f"Import error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}"
        st.error(f"❌ {error_msg}")
        return None, None, error_msg

class StreamlitTravelApp:
    def __init__(self):
        self.chat_system = None
        self.import_error = None
        self.demo_mode = False
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the hybrid chat system with comprehensive error handling"""
        try:
            with st.spinner("🚀 Initializing AI Travel Assistant..."):
                HybridChatSystemClass, _, error = safe_import()
                
                if error:
                    self.import_error = error
                    self.demo_mode = True
                    st.warning("🔧 Running in limited demo mode. Some features may not be available.")
                    return
                
                # Try to create the chat system
                self.chat_system = HybridChatSystemClass()
                
                # Test if system is actually working
                status = self.chat_system.get_system_status()
                if status.get("pinecone_connected") and status.get("groq_configured"):
                    st.success("✅ AI System Ready!")
                else:
                    st.warning("⚠️ System partially initialized. Some features may be limited.")
                    
        except Exception as e:
            self.import_error = f"Failed to initialize: {str(e)}"
            self.demo_mode = True
            st.error(f"❌ {self.import_error}")
            st.info("💡 Running in demo mode with sample data")
    
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
        
        if self.demo_mode:
            st.warning("""
            **Demo Mode Active** 
            - Using sample data for demonstration
            - Full features require API configuration
            - Contact administrator for full access
            """)
    
    def display_sidebar(self):
        """Display sidebar with information and quick queries"""
        with st.sidebar:
            st.markdown("### ⚡ Quick Queries")
            
            quick_queries = [
                "Create a romantic 4 day itinerary for Vietnam",
                "Best beach destinations in Vietnam",
                "Adventure activities in northern Vietnam", 
                "Cultural and historical sites in central Vietnam",
                "Budget travel options in Vietnam"
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
            else:
                st.warning("🔧 System initializing...")
            
            st.markdown("---")
            st.markdown("""
            **💡 Powered by:**
            - 🗃️ **Pinecone** - Vector Search
            - 🕸️ **Neo4j** - Knowledge Graph  
            - 🤖 **Groq LLM** - AI Reasoning
            - 🔤 **Local Embeddings** - Semantic Search
            """)
    
    def get_demo_response(self, query):
        """Provide demo responses when APIs are not available"""
        demo_responses = {
            "romantic itinerary": """
**Romantic 4-Day Vietnam Itinerary** 🌹

**Day 1: Hanoi - Cultural Beginnings**
- Arrive in Hanoi, check into a boutique hotel in the Old Quarter
- Evening: Romantic dinner at a French colonial restaurant
- Stroll around Hoan Kiem Lake as the city lights reflect on the water

**Day 2: Hanoi to Hoi An**
- Morning flight to Da Nang, transfer to Hoi An
- Afternoon: Explore Hoi An's ancient town with its iconic lanterns
- Evening: Private lantern-lit boat ride on Thu Bon River

**Day 3: Hoi An Romance**
- Morning: Private cooking class for Vietnamese cuisine
- Afternoon: Beach time at An Bang Beach
- Evening: Romantic dinner at riverside restaurant

**Day 4: Da Lat - Mountain Escape**
- Flight to Da Lat, the "City of Eternal Spring"
- Visit flower gardens and romantic waterfalls
- Farewell dinner with mountain views

*Note: This is a sample itinerary. Full AI-powered recommendations require API configuration.*
            """,
            "default": """
I'd love to help you plan your Vietnam travel itinerary! 🏝️

Currently running in demonstration mode. For personalized AI-powered recommendations with real-time data from our travel database, please ensure:

1. **Pinecone API** is configured for destination search
2. **Groq API** is set up for AI responses
3. **Environment variables** are properly set

**Sample destinations you might consider:**
- 🏙️ **Hanoi**: Cultural capital with amazing street food
- 🏮 **Hoi An**: Ancient town with romantic lantern-lit streets  
- 🌸 **Da Lat**: Mountain retreat with beautiful flowers
- 🏖️ **Nha Trang**: Coastal city with stunning beaches

*Contact the administrator to enable full AI capabilities.*
            """
        }
        
        query_lower = query.lower()
        if "romantic" in query_lower and "itinerary" in query_lower:
            return demo_responses["romantic itinerary"]
        else:
            return demo_responses["default"]
    
    def display_demo_metrics(self):
        """Display demo metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>🔍 Vector Results</h3>
                <h2>5</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>🕸️ Location Results</h3>
                <h2>3</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>⏱️ Search Time</h3>
                <h2>1.2s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <h3>⚡ Total Time</h3>
                <h2>2.5s</h2>
            </div>
            """, unsafe_allow_html=True)
    
    def display_demo_results(self):
        """Display demo search results"""
        st.markdown("### 📊 Search Results")
        
        # Demo vector results
        with st.expander("🗃️ Vector Search Results (5 found)", expanded=True):
            demo_vector_results = [
                {"name": "Hanoi", "region": "Northern Vietnam", "type": "City", "description": "Cultural capital with rich history and amazing street food experiences.", "tags": ["culture", "food", "heritage"], "best_time": "Feb-May", "score": 0.85},
                {"name": "Hoi An", "region": "Central Vietnam", "type": "City", "description": "Ancient town famous for lantern-lit streets and romantic riverside atmosphere.", "tags": ["lanterns", "romantic", "heritage"], "best_time": "Oct-Apr", "score": 0.82},
                {"name": "Da Lat", "region": "Southern Vietnam", "type": "City", "description": "Mountain retreat known as the City of Eternal Spring with beautiful flowers.", "tags": ["mountain", "romantic", "flowers"], "best_time": "Feb-May", "score": 0.78},
            ]
            
            for i, result in enumerate(demo_vector_results, 1):
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="margin: 0; color: #1f77b4;">{i}. {result['name']}</h4>
                    <p style="margin: 0.25rem 0; color: #666;">
                        📍 <strong>Location:</strong> {result['region']} | 
                        🏷️ <strong>Type:</strong> {result['type']}
                    </p>
                    <p style="margin: 0.5rem 0; color: #333;">{result['description']}</p>
                    <p style="margin: 0.25rem 0; color: #555;">
                        🎯 <strong>Tags:</strong> {', '.join(result['tags'])}
                    </p>
                    <p style="margin: 0.25rem 0; color: #555;">
                        📅 <strong>Best Time:</strong> {result['best_time']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
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
            
            if self.demo_mode or not self.chat_system:
                # Use demo mode
                with st.spinner("💫 Generating demo response..."):
                    time.sleep(2)  # Simulate processing time
                    response = self.get_demo_response(user_input)
                    
                    st.session_state.last_response = {
                        'query': user_input,
                        'response': response,
                        'demo_mode': True
                    }
            else:
                # Use real AI system
                with st.spinner("🔍 Searching travel database and generating response..."):
                    try:
                        pinecone_results, neo4j_results, response, total_time = asyncio.run(
                            self.chat_system.process_query_with_metrics(user_input)
                        )
                        
                        st.session_state.last_response = {
                            'query': user_input,
                            'response': response,
                            'pinecone_results': pinecone_results,
                            'neo4j_results': neo4j_results,
                            'demo_mode': False
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                        # Fallback to demo mode
                        response = self.get_demo_response(user_input)
                        st.session_state.last_response = {
                            'query': user_input,
                            'response': response,
                            'demo_mode': True
                        }
        
        # Display results if available
        if st.session_state.last_response:
            data = st.session_state.last_response
            
            if data.get('demo_mode', False):
                # Display demo results
                self.display_demo_metrics()
                self.display_demo_results()
            else:
                # Display real results (you can add your existing display logic here)
                pass
            
            # Display AI response
            st.markdown("### 🧠 Travel Assistant Response")
            st.markdown(
                f"""
                <div class="response-container">
                    <div style="color: #2c3e50; font-size: 16px; line-height: 1.7;">
                        {data['response']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    # Initialize and run the app
    app = StreamlitTravelApp()
    app.run()

if __name__ == "__main__":
    main()

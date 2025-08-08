"""
Simplified Streamlit App for Sales Forecasting AI
This version works with the simplified modules for testing purposes.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="env_example.txt")
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings_simple import SimpleEmbeddingManager
from retrieval_simple import SimpleRAGSystem

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .response-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching."""
    # Initialize RAG system without embedding manager
    rag_system = SimpleRAGSystem()
    
    return rag_system

def load_sample_data():
    """Load sample sales data."""
    try:
        df = pd.read_csv("data/sample_sales_data.csv")
        return df
    except FileNotFoundError:
        st.error("Sample data not found. Please run data ingestion first.")
        return None

def create_sales_overview(df):
    """Create sales overview visualizations."""
    if df is None:
        return
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Monthly sales trend
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_sales['date'] = monthly_sales['date'].astype(str)
    
    fig_monthly = px.line(
        monthly_sales, 
        x='date', 
        y='sales',
        title="Monthly Sales Trend",
        labels={'sales': 'Sales Amount ($)', 'date': 'Month'}
    )
    fig_monthly.update_layout(height=400)
    
    # Product performance
    product_sales = df.groupby('product')['sales'].sum().sort_values(ascending=False)
    fig_product = px.bar(
        x=product_sales.index,
        y=product_sales.values,
        title="Product Performance",
        labels={'x': 'Product', 'y': 'Total Sales ($)'}
    )
    fig_product.update_layout(height=400)
    
    # Region performance
    region_sales = df.groupby('region')['sales'].sum().sort_values(ascending=False)
    fig_region = px.pie(
        values=region_sales.values,
        names=region_sales.index,
        title="Sales by Region"
    )
    fig_region.update_layout(height=400)
    
    return fig_monthly, fig_product, fig_region

def display_rag_response(response):
    """Display RAG response."""
    if response.get('success', False):
        st.success("✅ AI Response Generated")
        
        # Show only the answer in a clean format
        st.subheader("🤖 AI Answer")
        st.write(response['answer'])
        
        # Optional: Show technical details in an expander (collapsed by default)
        with st.expander("🔧 Technical Details (Click to see)"):
            st.write(f"**Model:** {response.get('model_used', 'Unknown')}")
            st.write(f"**Query Type:** {response.get('query_type', 'Unknown')}")
            st.write(f"**Chunks Used:** {response.get('chunks_retrieved', 0)}")
            if response.get('tokens_used'):
                st.write(f"**Tokens Used:** {response['tokens_used']}")
    else:
        st.error("❌ Failed to generate response")
        st.write(response.get('answer', 'Unknown error'))

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">📊 Sales Forecasting AI</h1>', unsafe_allow_html=True)
    
    # Initialize system
    try:
        rag_system = initialize_system()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "AI Assistant", "Data Analysis", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard(rag_system)
    elif page == "AI Assistant":
        show_ai_assistant(rag_system)
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Settings":
        show_settings(rag_system)

def show_dashboard(rag_system):
    """Show the main dashboard."""
    st.header("📈 Sales Dashboard")
    
    # Load sample data
    df = load_sample_data()
    
    if df is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
        with col2:
            st.metric("Average Sale", f"${df['sales'].mean():.2f}")
        with col3:
            st.metric("Total Orders", len(df))
        with col4:
            st.metric("Products", df['product'].nunique())
        
        # Visualizations
        st.subheader("📊 Sales Analytics")
        
        fig_monthly, fig_product, fig_region = create_sales_overview(df)
        
        if fig_monthly and fig_product and fig_region:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_monthly, use_container_width=True)
                st.plotly_chart(fig_product, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_region, use_container_width=True)
                
                # Recent activity
                st.subheader("🕒 Recent Activity")
                recent_sales = df.sort_values('date', ascending=False).head(10)
                st.dataframe(recent_sales[['date', 'product', 'region', 'sales']])

def show_ai_assistant(rag_system):
    """Show the AI assistant interface."""
    st.header("🤖 AI Sales Assistant")
    
    st.markdown("""
    Ask questions about your sales data, trends, and forecasts. The AI will analyze your data and provide insights.
    """)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What were the sales trends in Q1? Can you forecast next quarter's sales?",
        height=100
    )
    
    # Ask button
    if st.button("🚀 Ask AI", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🤖 AI is analyzing your data..."):
                    response = rag_system.answer_query(query, 5)  # Default to 5 chunks
                    display_rag_response(response)
                    
                    # Optional: Show retrieved chunks in an expander (collapsed by default)
                    if response.get('chunks_retrieved', 0) > 0:
                        with st.expander("📄 Data Sources Used (Click to see)"):
                            for i, chunk in enumerate(response['chunks'][:3]):
                                st.text(f"Source {i+1}: {chunk['document'][:100]}...")
            else:
                st.warning("Please enter a question.")
    
    # Example queries
    st.subheader("💡 Example Questions")
    example_queries = [
        "What were the sales trends in the last quarter?",
        "Can you forecast sales for the next month?",
        "Which products are performing best?",
        "What is the overall sales performance?",
        "Show me trend analysis for the last year"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"Example {i+1}: {example}", key=f"example_{i}"):
            st.session_state.query = example
            st.rerun()

def show_data_analysis():
    """Show data analysis page."""
    st.header("📋 Data Analysis")
    
    df = load_sample_data()
    
    if df is not None:
        st.subheader("📊 Data Overview")
        
        # Data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Total rows: {len(df)}")
            st.write(f"- Date range: {df['date'].min()} to {df['date'].max()}")
            st.write(f"- Products: {df['product'].nunique()}")
            st.write(f"- Regions: {df['region'].nunique()}")
        
        with col2:
            st.write("**Summary Statistics:**")
            st.write(f"- Mean sales: ${df['sales'].mean():.2f}")
            st.write(f"- Median sales: ${df['sales'].median():.2f}")
            st.write(f"- Max sales: ${df['sales'].max():.2f}")
            st.write(f"- Min sales: ${df['sales'].min():.2f}")
        
        # Raw data
        st.subheader("📄 Raw Data")
        st.dataframe(df, use_container_width=True)

def show_settings(rag_system):
    """Show settings page."""
    st.header("⚙️ Settings")
    
    # System information
    st.subheader("System Information")
    
    stats = rag_system.get_system_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Configuration:**")
        st.write(f"- Model Type: {stats['model_type']}")
        st.write(f"- Max Context Length: {stats['max_context_length']}")
        st.write(f"- Embedding Manager: {'✅ Available' if stats['embedding_manager_available'] else '❌ Not Available'}")
    
    with col2:
        st.write("**Available Features:**")
        st.write(f"- Prompt Templates: {len(stats['available_prompt_templates'])}")
        for template in stats['available_prompt_templates']:
            st.write(f"  - {template}")
    
    # Embeddings info
    st.subheader("📚 Embeddings Information")
    
    embeddings_file = Path("models/chroma_db/embeddings.json")
    if embeddings_file.exists():
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        st.write(f"- Total embeddings: {len(embeddings_data['embeddings'])}")
        st.write(f"- Embedding dimensions: {len(embeddings_data['embeddings'][0]) if embeddings_data['embeddings'] else 0}")
    else:
        st.warning("No embeddings found. Please run data processing first.")

if __name__ == "__main__":
    main()

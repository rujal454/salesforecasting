#!/usr/bin/env python3
"""
Setup script for Real AI Sales Forecasting
This script helps you configure real AI and use your own CSV data.
"""

import os
import sys
import shutil
from pathlib import Path

def check_openai_setup():
    """Check if OpenAI is properly configured."""
    print("🔍 Checking OpenAI Setup...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Creating .env file...")
        
        api_key = input("Enter your OpenAI API key (get it from https://platform.openai.com/api-keys): ").strip()
        
        if api_key:
            with open(".env", "w") as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            print("✅ .env file created with your API key")
        else:
            print("❌ No API key provided. Please create .env file manually.")
            return False
    else:
        print("✅ .env file found")
    
    # Check if openai package is installed
    try:
        import openai
        print("✅ OpenAI package installed")
    except ImportError:
        print("❌ OpenAI package not installed")
        print("Installing openai package...")
        os.system("pip install openai")
        print("✅ OpenAI package installed")
    
    return True

def setup_csv_data():
    """Help user set up their CSV data."""
    print("\n📊 Setting up CSV Data...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # List existing CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if csv_files:
        print("📁 Found existing CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {file.name}")
    
    # Ask user to add their CSV
    print("\n📋 To use your own CSV data:")
    print("1. Copy your CSV file to the 'data' folder")
    print("2. Make sure it has these columns: date, product, region, quantity, price, sales")
    print("3. Or update the code to match your column names")
    
    csv_path = input("\nEnter path to your CSV file (or press Enter to skip): ").strip()
    
    if csv_path and Path(csv_path).exists():
        target_path = data_dir / Path(csv_path).name
        shutil.copy2(csv_path, target_path)
        print(f"✅ Copied {csv_path} to {target_path}")
        return target_path.name
    else:
        print("ℹ️  Using sample data for now")
        return "sample_sales_data.csv"

def create_real_ai_app():
    """Create a Streamlit app that uses real AI."""
    print("\n🤖 Creating Real AI App...")
    
    app_content = '''"""
Real AI Sales Forecasting App
This version uses OpenAI GPT for intelligent responses.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from retrieval_real_ai import RealAISystem

# Page configuration
st.set_page_config(
    page_title="Real AI Sales Forecasting",
    page_icon="🚀",
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
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the real AI system."""
    try:
        rag_system = RealAISystem()
        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize AI system: {e}")
        st.info("Please check your OpenAI API key in .env file")
        return None

def load_csv_data(csv_file="sample_sales_data.csv"):
    """Load CSV data."""
    try:
        data_path = Path("data") / csv_file
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            st.error(f"CSV file not found: {data_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sales_overview(df):
    """Create sales overview visualizations."""
    try:
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
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None, None, None

def display_rag_response(response):
    """Display RAG response."""
    if response.get('success', False):
        st.success("✅ AI Response Generated")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🤖 AI Answer")
            st.write(response['answer'])
        
        with col2:
            st.subheader("📊 Response Info")
            st.write(f"**Model:** {response.get('model_used', 'Unknown')}")
            st.write(f"**Query Type:** {response.get('query_type', 'Unknown')}")
            st.write(f"**Chunks Used:** {response.get('chunks_retrieved', 0)}")
            if response.get('tokens_used'):
                st.write(f"**Tokens Used:** {response['tokens_used']}")
    else:
        st.error("❌ Failed to generate response")
        st.write(response.get('answer', 'Unknown error'))

def main():
    """Main application."""
    st.markdown('<h1 class="main-header">🚀 Real AI Sales Forecasting</h1>', unsafe_allow_html=True)
    
    # Initialize system
    rag_system = initialize_system()
    
    if not rag_system:
        st.error("❌ AI system not initialized. Please check your OpenAI API key.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # CSV file selection
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    csv_options = [f.name for f in csv_files] if csv_files else ["sample_sales_data.csv"]
    
    selected_csv = st.sidebar.selectbox(
        "📁 Select CSV File",
        csv_options,
        index=0
    )
    
    # Load data
    df = load_csv_data(selected_csv)
    
    if df is None:
        st.error("❌ No data loaded. Please check your CSV file.")
        return
    
    # Main navigation
    page = st.sidebar.selectbox(
        "📄 Navigation",
        ["Dashboard", "AI Assistant", "Data Analysis", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard(rag_system, df)
    elif page == "AI Assistant":
        show_ai_assistant(rag_system, df)
    elif page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Settings":
        show_settings(rag_system)

def show_dashboard(rag_system, df):
    """Show the dashboard."""
    st.header("📊 Sales Dashboard")
    
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

def show_ai_assistant(rag_system, df):
    """Show the AI assistant interface."""
    st.header("🤖 Real AI Sales Assistant")
    
    st.markdown("""
    This AI assistant uses **OpenAI GPT** to provide intelligent insights about your sales data.
    Ask questions about trends, forecasts, and analysis.
    """)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the top performing products? Can you forecast next quarter's sales?",
        height=100
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        n_results = st.slider("Number of chunks to retrieve", 1, 10, 5)
    with col2:
        if st.button("🚀 Ask Real AI", type="primary"):
            if query.strip():
                with st.spinner("🤖 AI is analyzing your data..."):
                    response = rag_system.answer_query(query, n_results)
                    display_rag_response(response)
                    
                    # Show retrieved chunks
                    if response.get('chunks_retrieved', 0) > 0:
                        st.subheader("📄 Retrieved Context")
                        for i, chunk in enumerate(response['chunks'][:3]):
                            with st.expander(f"Chunk {i+1} (Similarity: {chunk.get('similarity', 0):.3f})"):
                                st.text(chunk['document'][:200] + "...")
            else:
                st.warning("Please enter a question.")
    
    # Example queries
    st.subheader("💡 Example Questions")
    example_queries = [
        "What are the top 5 products by sales?",
        "Can you forecast sales for the next quarter?",
        "Which region has the highest sales growth?",
        "What are the sales trends by month?",
        "Show me a detailed analysis of product performance"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"Example {i+1}: {example}", key=f"example_{i}"):
            st.session_state.query = example
            st.rerun()

def show_data_analysis(df):
    """Show data analysis page."""
    st.header("📋 Data Analysis")
    
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
    st.dataframe(df.head(20))

def show_settings(rag_system):
    """Show settings page."""
    st.header("⚙️ Settings")
    
    # System stats
    stats = rag_system.get_system_stats()
    
    st.subheader("🤖 AI System Status")
    st.write(f"**Model:** {stats['model']}")
    st.write(f"**OpenAI Configured:** {'✅ Yes' if stats['openai_configured'] else '❌ No'}")
    st.write(f"**Max Context Length:** {stats['max_context_length']}")
    st.write(f"**Available Templates:** {', '.join(stats['prompt_templates'])}")
    
    if not stats['openai_configured']:
        st.error("❌ OpenAI not configured. Please add your API key to .env file")
        st.info("Get your API key from: https://platform.openai.com/api-keys")

if __name__ == "__main__":
    main()
'''
    
    with open("ui/streamlit_app_real_ai.py", "w") as f:
        f.write(app_content)
    
    print("✅ Real AI app created: ui/streamlit_app_real_ai.py")

def main():
    """Main setup function."""
    print("🚀 Real AI Sales Forecasting Setup")
    print("=" * 50)
    
    # Check OpenAI setup
    if not check_openai_setup():
        print("\n❌ OpenAI setup failed. Please configure your API key.")
        return
    
    # Setup CSV data
    csv_file = setup_csv_data()
    
    # Create real AI app
    create_real_ai_app()
    
    print("\n✅ Setup Complete!")
    print("\n🎯 Next Steps:")
    print("1. Run the real AI app: streamlit run ui/streamlit_app_real_ai.py")
    print("2. Add your CSV file to the 'data' folder")
    print("3. Ask intelligent questions about your sales data!")
    
    print(f"\n📊 Current data file: {csv_file}")
    print("🤖 AI Model: OpenAI GPT-3.5-turbo")
    print("💡 Cost: ~$0.002 per request")

if __name__ == "__main__":
    main()

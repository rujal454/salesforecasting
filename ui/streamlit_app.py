"""
Phase 4: Streamlit UI
Build Streamlit app with natural language query interface, visualizations, and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="env_example.txt")  # explicitly load env.txt
print("LOADED KEY:", os.getenv("OPENAI_API_KEY"))

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_ingestion import DataIngestion
from embeddings import EmbeddingManager
from retrieval import RAGSystem
from forecasting import SalesForecaster

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
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_systems():
    """Initialize all systems with caching"""
    try:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(embedding_model="sbert")
        
        # Initialize RAG system
        rag_system = RAGSystem(embedding_manager=embedding_manager)
        
        # Initialize forecaster
        forecaster = SalesForecaster()
        
        return embedding_manager, rag_system, forecaster
    except Exception as e:
        st.error(f"Error initializing systems: {e}")
        return None, None, None

def load_sample_data():
    """Load or create sample data"""
    data_file = Path("data/sample_sales_data.csv")
    if data_file.exists():
        return pd.read_csv(data_file)
    else:
        # Create sample data if it doesn't exist
        ingestion = DataIngestion()
        ingestion.create_sample_data()
        return pd.read_csv(data_file)

def create_sales_overview(df):
    """Create sales overview visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly sales trend
        df['date'] = pd.to_datetime(df['date'])
        monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].astype(str)
        
        fig_monthly = px.line(monthly_sales, x='date', y='sales', 
                             title="Monthly Sales Trend",
                             labels={'sales': 'Total Sales ($)', 'date': 'Month'})
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Product performance
        product_sales = df.groupby('product')['sales'].sum().sort_values(ascending=True)
        
        fig_products = px.bar(x=product_sales.values, y=product_sales.index, 
                             orientation='h',
                             title="Product Performance",
                             labels={'x': 'Total Sales ($)', 'y': 'Product'})
        fig_products.update_layout(height=400)
        st.plotly_chart(fig_products, use_container_width=True)
    
    # Regional performance
    st.subheader("Regional Performance")
    col3, col4 = st.columns(2)
    
    with col3:
        region_sales = df.groupby('region')['sales'].sum()
        fig_region = px.pie(values=region_sales.values, names=region_sales.index,
                           title="Sales by Region")
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col4:
        # Sales heatmap by month and region
        df['month'] = df['date'].dt.month
        heatmap_data = df.groupby(['month', 'region'])['sales'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='region', columns='month', values='sales')
        
        fig_heatmap = px.imshow(heatmap_pivot, 
                               title="Sales Heatmap: Month vs Region",
                               labels=dict(x="Month", y="Region", color="Sales"))
        st.plotly_chart(fig_heatmap, use_container_width=True)

def display_rag_response(response):
    """Display RAG response with supporting data"""
    st.subheader("🤖 AI Response")
    
    # Display the main answer
    st.markdown(f"**Answer:** {response['answer']}")
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Retrieved", response['chunks_retrieved'])
    with col2:
        st.metric("Query Type", response['query_type'])
    with col3:
        if response.get('tokens_used'):
            st.metric("Tokens Used", response['tokens_used'])
    
    # Display supporting chunks
    if response.get('supporting_chunks'):
        with st.expander("📋 Supporting Data"):
            for i, chunk in enumerate(response['supporting_chunks']):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(f"Period: {chunk['metadata'].get('period', 'Unknown')}")
                st.markdown(f"Type: {chunk['metadata'].get('chunk_type', 'Unknown')}")
                st.markdown(f"Content: {chunk['text'][:200]}...")
                st.divider()

def display_forecast_results(forecast_report):
    """Display forecasting results"""
    st.subheader("📈 Forecasting Results")
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Used", forecast_report['model_used'].title())
    with col2:
        st.metric("R² Score", f"{forecast_report['model_metrics']['r2_test']:.4f}")
    with col3:
        st.metric("RMSE", f"{forecast_report['model_metrics']['rmse_test']:.2f}")
    with col4:
        st.metric("Training Points", forecast_report['training_data_points'])
    
    # Forecast statistics
    st.subheader("Forecast Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Prediction", f"${forecast_report['forecast_stats']['mean_prediction']:,.2f}")
    with col2:
        st.metric("Total Predicted", f"${forecast_report['forecast_stats']['total_predicted_sales']:,.2f}")
    with col3:
        st.metric("Min Prediction", f"${forecast_report['forecast_stats']['min_prediction']:,.2f}")
    with col4:
        st.metric("Max Prediction", f"${forecast_report['forecast_stats']['max_prediction']:,.2f}")
    
    # Feature importance
    if forecast_report.get('feature_importance'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame(
            list(forecast_report['feature_importance'].items()),
            columns=['Feature', 'Importance']
        ).head(10)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h',
                               title="Top 10 Most Important Features")
        st.plotly_chart(fig_importance, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">📊 Sales Forecasting AI</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    with st.spinner("Initializing AI systems..."):
        embedding_manager, rag_system, forecaster = initialize_systems()
    
    if not all([embedding_manager, rag_system, forecaster]):
        st.error("Failed to initialize systems. Please check your configuration.")
        return
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🤖 AI Assistant", "📈 Forecasting", "📊 Data Analysis", "⚙️ Settings"]
    )
    
    # Load data
    try:
        df = load_sample_data()
        st.sidebar.success(f"✅ Loaded {len(df):,} sales records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(df, rag_system)
    elif page == "🤖 AI Assistant":
        show_ai_assistant(rag_system)
    elif page == "📈 Forecasting":
        show_forecasting(df, forecaster)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "⚙️ Settings":
        show_settings(embedding_manager, rag_system)

def show_dashboard(df, rag_system):
    """Show the main dashboard"""
    st.header("🏠 Sales Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sales", f"${df['sales'].sum():,.2f}")
    with col2:
        st.metric("Average Daily Sales", f"${df['sales'].mean():,.2f}")
    with col3:
        st.metric("Total Products", df['product'].nunique())
    with col4:
        st.metric("Total Regions", df['region'].nunique())
    
    # Sales overview
    st.subheader("Sales Overview")
    create_sales_overview(df)
    
    # Quick AI insights
    st.subheader("🤖 Quick AI Insights")
    quick_questions = [
        "What are the top 3 performing products?",
        "Which region has the highest sales?",
        "What is the average sales trend?"
    ]
    
    selected_question = st.selectbox("Choose a quick question:", quick_questions)
    
    if st.button("Get AI Insight"):
        with st.spinner("Analyzing..."):
            response = rag_system.answer_query(selected_question)
            display_rag_response(response)

def show_ai_assistant(rag_system):
    """Show the AI assistant interface"""
    st.header("🤖 AI Sales Assistant")
    
    st.markdown("""
    Ask me anything about your sales data! I can help you with:
    - **Analysis**: "What were the top performing products in Q3?"
    - **Trends**: "Show me sales trends for the Northeast region"
    - **Forecasting**: "Predict sales for next quarter"
    - **Insights**: "Which products have declining sales trends?"
    """)
    
    # Query input
    query = st.text_area("Enter your question:", height=100, 
                        placeholder="e.g., What were the top performing products in Q3 2023?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        n_results = st.slider("Number of supporting chunks:", 3, 10, 5)
    with col2:
        if st.button("🔍 Ask AI", type="primary"):
            if query.strip():
                with st.spinner("Processing your question..."):
                    response = rag_system.answer_query(query, n_results)
                    display_rag_response(response)
            else:
                st.warning("Please enter a question.")
    
    # System stats
    with st.expander("📊 System Information"):
        stats = rag_system.get_system_stats()
        if 'error' not in stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats['total_documents'])
            with col2:
                st.metric("LLM Model", stats['model_used'])
            with col3:
                st.metric("Embedding Model", stats['embedding_model'])

def show_forecasting(df, forecaster):
    """Show the forecasting interface"""
    st.header("📈 Sales Forecasting")
    
    st.markdown("""
    Generate comprehensive sales forecasts using machine learning models.
    The system will automatically choose the best model based on performance.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_periods = st.slider("Forecast Period (days):", 30, 365, 90)
    with col2:
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Training models and generating forecast..."):
                try:
                    forecast_report = forecaster.generate_forecast_report(df, forecast_periods)
                    
                    # Save report
                    forecaster.save_forecast_report(forecast_report, "models/forecast_report.json")
                    
                    # Display results
                    display_forecast_results(forecast_report)
                    
                    # Create and display forecast plot
                    predictions_df = pd.DataFrame(forecast_report['predictions'])
                    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
                    
                    fig = forecaster.plot_forecast(df, predictions_df, "Sales Forecast")
                    st.pyplot(fig)
                    
                    st.success("✅ Forecast generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

def show_data_analysis(df):
    """Show detailed data analysis"""
    st.header("📊 Data Analysis")
    
    # Data overview
    st.subheader("Data Overview")
    st.dataframe(df.head(10))
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(correlation_matrix, 
                            title="Correlation Matrix",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Time series analysis
    st.subheader("Time Series Analysis")
    df['date'] = pd.to_datetime(df['date'])
    
    # Daily sales with trend
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    fig_trend = px.line(daily_sales, x='date', y='sales',
                        title="Daily Sales with Trend",
                        labels={'sales': 'Daily Sales ($)', 'date': 'Date'})
    
    # Add trend line
    z = np.polyfit(range(len(daily_sales)), daily_sales['sales'], 1)
    p = np.poly1d(z)
    fig_trend.add_scatter(x=daily_sales['date'], y=p(range(len(daily_sales))),
                          mode='lines', name='Trend Line')
    
    st.plotly_chart(fig_trend, use_container_width=True)

def show_settings(embedding_manager, rag_system):
    """Show settings and configuration"""
    st.header("⚙️ Settings")
    
    st.subheader("System Configuration")
    
    # Model settings
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Embedding Model", embedding_manager.embedding_model)
        st.metric("LLM Model", rag_system.llm_model)
    with col2:
        st.metric("Max Context Length", rag_system.max_context_length)
        st.metric("ChromaDB Location", str(embedding_manager.chroma_persist_dir))
    
    # System information
    st.subheader("System Information")
    stats = rag_system.get_system_stats()
    if 'error' not in stats:
        st.json(stats)
    else:
        st.error(f"Error getting system stats: {stats['error']}")
    
    # Data management
    st.subheader("Data Management")
    if st.button("🔄 Refresh Embeddings"):
        with st.spinner("Refreshing embeddings..."):
            try:
                embedding_manager.process_all_data()
                st.success("✅ Embeddings refreshed successfully!")
            except Exception as e:
                st.error(f"Error refreshing embeddings: {e}")

if __name__ == "__main__":
    main() 
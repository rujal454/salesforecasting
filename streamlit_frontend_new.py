"""
Streamlit Frontend for Sales Forecasting AI
Reengineered with OpenRouter, Prophet, and ChromaDB integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time
import io
from typing import Dict, Any, Optional

# Optional encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# Configuration
BACKEND_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting AI - Reengineered",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .ai-response {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .forecast-card {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if backend is running and get service status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def upload_csv(file):
    """Upload CSV file to backend"""
    try:
        files = {'file': file}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Upload failed')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def generate_forecast(session_id, periods=30):
    """Generate forecast using Prophet backend"""
    try:
        data = {
            'session_id': session_id,
            'periods': periods
        }
        response = requests.post(f"{BACKEND_URL}/forecast", json=data, timeout=120)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Forecast failed')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def ask_question(session_id, question):
    """Ask AI question using OpenRouter backend"""
    try:
        data = {
            'session_id': session_id,
            'question': question
        }
        response = requests.post(f"{BACKEND_URL}/ask", json=data, timeout=90)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Question answering failed')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def get_session_data(session_id):
    """Get session data from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/session/{session_id}", timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Failed to get session data')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def perform_vector_search(query, n_results=5):
    """Perform semantic search using ChromaDB"""
    try:
        data = {
            'query': query,
            'n_results': n_results
        }
        response = requests.post(f"{BACKEND_URL}/vector-search", json=data, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Vector search failed')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def get_vector_stats():
    """Get ChromaDB collection statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/vector-stats", timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get('error', 'Failed to get vector stats')
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def detect_encoding_from_bytes(data: bytes, default: str = 'utf-8') -> str:
    """Detect encoding from raw bytes using chardet, fallback to default."""
    if HAS_CHARDET and data:
        try:
            result = chardet.detect(data)
            enc = result.get('encoding') or default
            return enc
        except Exception:
            return default
    return default

def read_csv_preview(uploaded_file):
    """Read uploaded CSV for preview with robust encoding fallbacks."""
    file_bytes = uploaded_file.getvalue()
    encodings_to_try = []
    detected = detect_encoding_from_bytes(file_bytes, default='utf-8')
    encodings_to_try.append(detected)
    for enc in ['utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError as ue:
            last_error = ue
            continue
        except Exception as e:
            last_error = e
            continue
    raise last_error or Exception("Failed to read CSV for preview")

def create_sales_chart(df, title="Sales Data"):
    """Create sales chart using Plotly"""
    try:
        # Ensure date column exists
        if 'date' not in df.columns:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                df['date'] = pd.to_datetime(df[date_cols[0]])
            else:
                # Create synthetic dates
                df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        
        # Ensure sales column exists
        if 'sales' not in df.columns:
            sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
            if sales_cols:
                df['sales'] = df[sales_cols[0]]
            else:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df['sales'] = df[numeric_cols[0]]
                else:
                    return None
        
        # Aggregate by date
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('date')
        
        fig = px.line(
            daily_sales,
            x='date',
            y='sales',
            title=title,
            labels={'sales': 'Sales Amount ($)', 'date': 'Date'}
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def create_forecast_chart(historical_data, forecast_data, title="Sales Forecast"):
    """Create forecast chart with historical and predicted data using Prophet output"""
    try:
        # Prepare historical data
        if isinstance(historical_data, list):
            hist_df = pd.DataFrame(historical_data)
        else:
            hist_df = historical_data
        
        # Prepare forecast data
        if isinstance(forecast_data, list):
            forecast_df = pd.DataFrame(forecast_data)
        else:
            forecast_df = forecast_data
        
        # Create combined chart
        fig = go.Figure()
        
        # Historical data
        if 'date' in hist_df.columns and 'sales' in hist_df.columns:
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            daily_hist = hist_df.groupby('date')['sales'].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=daily_hist['date'],
                y=daily_hist['sales'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
        
        # Forecast data with confidence intervals
        if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # Confidence intervals if available
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='red', width=1, dash='dot'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['lower_bound'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Confidence Interval',
                    line=dict(color='red', width=1, dash='dot'),
                    showlegend=False
                ))
        
        fig.update_layout(
            title=title,
            height=500,
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating forecast chart: {e}")
        return None

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🚀 Sales Forecasting AI - Reengineered</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by OpenRouter AI, Prophet Forecasting, and ChromaDB Vector Store</p>', unsafe_allow_html=True)
    
    # Check backend health
    backend_healthy, health_data = check_backend_health()
    
    if not backend_healthy:
        st.error("❌ Backend not available")
        st.info(f"Please start the Flask backend: `python app_new.py`")
        st.info(f"Backend should be running at: {BACKEND_URL}")
        return
    
    # Success message
    st.success("✅ Backend connected successfully!")
    
    # Sidebar
    st.sidebar.title("⚙️ System Status")
    
    # Service status
    if health_data and 'services' in health_data:
        services = health_data['services']
        
        # AI Service Status
        if services.get('ai_service'):
            st.sidebar.success(f"🤖 AI Service: ✅ {health_data.get('ai_model', 'OpenRouter')}")
        else:
            st.sidebar.error("🤖 AI Service: ❌ Not Available")
        
        # Forecasting Service Status
        if services.get('forecasting_service'):
            st.sidebar.success("📊 Forecasting Service: ✅ Prophet")
        else:
            st.sidebar.error("📊 Forecasting Service: ❌ Not Available")
        
        # Vector Store Status
        if services.get('vector_store'):
            st.sidebar.success("🔍 Vector Store: ✅ ChromaDB")
            # Show vector store stats
            if st.sidebar.button("📈 Vector Store Stats"):
                success, stats = get_vector_stats()
                if success:
                    st.sidebar.info(f"Documents: {stats.get('total_documents', 0)}")
                else:
                    st.sidebar.error(f"Stats error: {stats}")
        else:
            st.sidebar.error("🔍 Vector Store: ❌ Not Available")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "📄 Navigation",
        ["Upload Data", "Dashboard", "AI Assistant", "Vector Search", "Settings"]
    )
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    
    if page == "Upload Data":
        show_upload_page()
    elif page == "Dashboard":
        show_dashboard_page()
    elif page == "AI Assistant":
        show_ai_assistant_page()
    elif page == "Vector Search":
        show_vector_search_page()
    elif page == "Settings":
        show_settings_page()

def show_upload_page():
    """Show file upload page"""
    st.header("📁 Upload Sales Data")
    
    st.markdown("""
    Upload your CSV file containing sales data. The file should have columns like:
    - `date` (or similar date column)
    - `sales` (or similar sales/revenue column)
    - Additional columns like `product`, `region`, `quantity`, etc.
    
    **Supported formats:** CSV, Excel (.xlsx, .xls)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with your sales data"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.info(f"📄 File: {uploaded_file.name}")
        st.info(f"📏 Size: {uploaded_file.size} bytes")
        
        # Preview data with robust encoding handling
        try:
            if uploaded_file.name.endswith('.csv'):
                df = read_csv_preview(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.info(f"📅 Columns: {', '.join(df.columns.tolist())}")
            
            # Upload button
            if st.button("🚀 Upload to Backend", type="primary"):
                with st.spinner("Uploading file..."):
                    # Rewind file pointer for upload
                    uploaded_file.seek(0)
                    success, result = upload_csv(uploaded_file)
                    
                    if success:
                        st.session_state.session_id = result['session_id']
                        st.session_state.uploaded_data = df
                        
                        st.success("✅ File uploaded successfully!")
                        st.info(f"Session ID: {result['session_id']}")
                        st.info(f"Rows: {result['rows']}")
                        st.info(f"Columns: {', '.join(result['columns'])}")
                        
                        if result.get('embeddings_stored'):
                            st.success("✅ Data embeddings stored in ChromaDB")
                        
                        # Auto-navigate to dashboard
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {result}")
                        
        except Exception as e:
            st.error(f"❌ Error reading file for preview: {e}")

def show_dashboard_page():
    """Show dashboard page"""
    st.header("📊 Sales Dashboard")
    
    if not st.session_state.session_id:
        st.warning("⚠️ No data uploaded. Please upload a file first.")
        return
    
    # Get session data
    success, session_data = get_session_data(st.session_state.session_id)
    
    if not success:
        st.error(f"❌ Failed to get session data: {session_data}")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Session ID", st.session_state.session_id[:8] + "...")
    with col2:
        st.metric("Data Rows", session_data['data_shape'][0])
    with col3:
        st.metric("Data Columns", session_data['data_shape'][1])
    with col4:
        st.metric("Has Forecast", "✅ Yes" if session_data['has_forecast'] else "❌ No")
    
    # Data visualization
    if st.session_state.uploaded_data is not None:
        st.subheader("📈 Sales Data Visualization")
        
        # Create sales chart
        fig = create_sales_chart(st.session_state.uploaded_data, "Historical Sales Data")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast section
        st.subheader("🔮 Scikit-Learn Sales Forecast")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            periods = st.slider("Forecast Periods (days)", 7, 365, 30)
        
        with col2:
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner("Training scikit-learn model and generating forecast..."):
                    success, result = generate_forecast(st.session_state.session_id, periods)
                    
                    if success:
                        st.session_state.forecast_data = result['forecast']
                        
                        st.success("✅ Forecast generated successfully!")
                        
                        # Show scikit-learn metrics
                        metrics = result['training_metrics']
                        st.info(f"**Scikit-Learn Model Performance:**")
                        st.info(f"  • MAE: ${metrics['mae']:,.2f}")
                        st.info(f"  • RMSE: ${metrics['rmse']:,.2f}")
                        st.info(f"  • MAPE: {metrics['mape']:.2f}%")
                        st.info(f"  • Training samples: {metrics['training_samples']}")
                        
                        # Show model insights
                        if 'model_info' in result:
                            model_info = result['model_info']
                            st.info(f"**Model Insights:**")
                            st.info(f"  • Changepoints detected: {model_info.get('changepoints', 0)}")
                            st.info(f"  • Seasonalities: {', '.join(model_info.get('seasonalities', []))}")
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Forecast failed: {result}")
        
        # Show forecast chart if available
        if st.session_state.forecast_data:
            st.subheader("📊 Forecast Visualization")
            
            forecast_fig = create_forecast_chart(
                st.session_state.uploaded_data,
                st.session_state.forecast_data,
                "Scikit-Learn Sales Forecast"
            )
            
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Forecast summary
            forecast_df = pd.DataFrame(st.session_state.forecast_data)
            if 'forecast' in forecast_df.columns:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Forecast Periods", len(forecast_df))
                with col2:
                    st.metric("Avg Forecast", f"${forecast_df['forecast'].mean():,.2f}")
                with col3:
                    st.metric("Total Forecast", f"${forecast_df['forecast'].sum():,.2f}")
                with col4:
                    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                        confidence_range = forecast_df['upper_bound'].mean() - forecast_df['lower_bound'].mean()
                        st.metric("Avg Confidence Range", f"${confidence_range:.2f}")

def show_ai_assistant_page():
    """Show AI assistant page"""
    st.header("🤖 AI Sales Assistant (OpenRouter)")
    
    if not st.session_state.session_id:
        st.warning("⚠️ No data uploaded. Please upload a file first.")
        return
    
    st.markdown("""
    Ask questions about your sales data and get AI-powered insights using OpenRouter.
    The AI can analyze trends, provide forecasts, and answer complex questions about your data.
    """)
    
    # Question input
    question = st.text_area(
        "Ask a question about your sales data:",
        placeholder="e.g., What are the top performing products? Can you forecast next quarter's sales? Which region has the highest growth?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🤖 Ask AI", type="primary"):
            if question.strip():
                with st.spinner("🤖 AI is analyzing your data..."):
                    success, result = ask_question(st.session_state.session_id, question)
                    
                    if success:
                        st.success("✅ AI Response Generated")
                        
                        # Display answer
                        st.subheader("🤖 AI Answer")
                        st.markdown(f'<div class="ai-response">{result["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Model:** {result.get('model', 'Unknown')}")
                        with col2:
                            if 'tokens_used' in result:
                                st.info(f"**Tokens Used:** {result['tokens_used']}")
                        with col3:
                            st.info(f"**Context Used:** {'Yes' if result.get('context_used') else 'No'}")
                        
                        # Display context info
                        st.subheader("📊 Context Information")
                        context = result['context_summary']
                        st.info(f"Data rows: {context['data_rows']}")
                        st.info(f"Data columns: {context['data_columns']}")
                        st.info(f"Has forecast: {'Yes' if context['has_forecast'] else 'No'}")
                        
                    else:
                        st.error(f"❌ AI response failed: {result}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("💡 Example Questions")
        example_questions = [
            "What are the top 5 products by sales?",
            "Can you forecast sales for the next quarter?",
            "Which region has the highest sales growth?",
            "What are the sales trends by month?",
            "Show me a detailed analysis of product performance",
            "What factors are driving sales growth?",
            "Which products are declining in sales?",
            "What is the seasonal pattern in the data?"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.example_question = example
                st.rerun()
    
    # Auto-fill example question
    if 'example_question' in st.session_state:
        st.text_area("Selected example:", value=st.session_state.example_question, disabled=True)
        if st.button("Use this question"):
            question = st.session_state.example_question
            del st.session_state.example_question
            st.rerun()

def show_vector_search_page():
    """Show vector search page"""
    st.header("🔍 Semantic Vector Search (ChromaDB)")
    
    st.markdown("""
    Search through your stored data embeddings using natural language queries.
    This feature uses ChromaDB to find semantically similar content across your sessions.
    """)
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., sales trends, product performance, regional analysis"
        )
    
    with col2:
        n_results = st.number_input("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search", type="primary"):
        if search_query.strip():
            with st.spinner("Searching through embeddings..."):
                success, results = perform_vector_search(search_query, n_results)
                
                if success:
                    st.success(f"✅ Found {results['total_results']} results")
                    
                    # Display results
                    for i, result in enumerate(results['results']):
                        with st.expander(f"Result {i+1} (Relevance: {result['relevance_score']:.2f})"):
                            st.write(f"**Document:** {result['document']}")
                            st.write(f"**Metadata:** {result['metadata']}")
                            st.write(f"**Distance:** {result['distance']:.4f}")
                else:
                    st.error(f"❌ Search failed: {results}")
        else:
            st.warning("Please enter a search query.")
    
    # Show collection stats
    st.subheader("📊 Vector Store Statistics")
    if st.button("📈 Refresh Stats"):
        success, stats = get_vector_stats()
        if success:
            st.info(f"**Collection:** {stats.get('collection_name', 'Unknown')}")
            st.info(f"**Total Documents:** {stats.get('total_documents', 0)}")
            st.info(f"**Storage Directory:** {stats.get('persist_directory', 'Unknown')}")
        else:
            st.error(f"Failed to get stats: {stats}")

def show_settings_page():
    """Show settings page"""
    st.header("⚙️ System Settings")
    
    # Backend connection
    st.subheader("🔗 Backend Connection")
    
    backend_healthy, health_data = check_backend_health()
    
    if backend_healthy:
        st.success("✅ Backend connected")
        st.info(f"URL: {BACKEND_URL}")
        
        if health_data:
            st.info(f"Status: {health_data.get('status', 'Unknown')}")
            st.info(f"Timestamp: {health_data.get('timestamp', 'Unknown')}")
            
            # Service status
            if 'services' in health_data:
                services = health_data['services']
                st.subheader("🔧 Service Status")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if services.get('ai_service'):
                        st.success("🤖 AI Service: ✅ Available")
                    else:
                        st.error("🤖 AI Service: ❌ Not Available")
                
                with col2:
                    if services.get('forecasting_service'):
                        st.success("📊 Forecasting: ✅ Available")
                    else:
                        st.error("📊 Forecasting: ❌ Not Available")
                
                with col3:
                    if services.get('vector_store'):
                        st.success("🔍 Vector Store: ✅ Available")
                    else:
                        st.error("🔍 Vector Store: ❌ Not Available")
    else:
        st.error("❌ Backend not connected")
        st.info(f"Expected URL: {BACKEND_URL}")
    
    # Session information
    st.subheader("📋 Session Information")
    
    if st.session_state.session_id:
        st.info(f"Session ID: {st.session_state.session_id}")
        
        success, session_data = get_session_data(st.session_state.session_id)
        if success:
            st.info(f"Filename: {session_data['filename']}")
            st.info(f"Upload time: {session_data['upload_time']}")
            st.info(f"Data shape: {session_data['data_shape']}")
            st.info(f"Has forecast: {'Yes' if session_data['has_forecast'] else 'No'}")
        else:
            st.error(f"Failed to get session data: {session_data}")
    else:
        st.warning("No active session")
    
    # Clear session
    st.subheader("🗑️ Session Management")
    
    if st.button("Clear Session Data"):
        st.session_state.session_id = None
        st.session_state.uploaded_data = None
        st.session_state.forecast_data = None
        st.success("Session data cleared!")
        st.rerun()

if __name__ == "__main__":
    main()

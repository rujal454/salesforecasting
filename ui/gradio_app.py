"""
Phase 4: Gradio UI
Alternative UI using Gradio for the sales forecasting application
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_ingestion import DataIngestion
from embeddings import EmbeddingManager
from retrieval import RAGSystem
from forecasting import SalesForecaster

class GradioSalesForecasting:
    def __init__(self):
        self.embedding_manager = None
        self.rag_system = None
        self.forecaster = None
        self.df = None
        self.initialize_systems()
    
    def initialize_systems(self):
        """Initialize all systems"""
        try:
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(embedding_model="sbert")
            
            # Initialize RAG system
            self.rag_system = RAGSystem(embedding_manager=self.embedding_manager)
            
            # Initialize forecaster
            self.forecaster = SalesForecaster()
            
            # Load sample data
            self.load_sample_data()
            
            print("✅ Systems initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing systems: {e}")
    
    def load_sample_data(self):
        """Load or create sample data"""
        data_file = Path("data/sample_sales_data.csv")
        if data_file.exists():
            self.df = pd.read_csv(data_file)
        else:
            # Create sample data if it doesn't exist
            ingestion = DataIngestion()
            ingestion.create_sample_data()
            self.df = pd.read_csv(data_file)
    
    def answer_query(self, query, n_results):
        """Answer a user query using RAG"""
        if not query.strip():
            return "Please enter a question."
        
        try:
            response = self.rag_system.answer_query(query, int(n_results))
            return response['answer']
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def generate_forecast(self, forecast_days):
        """Generate sales forecast"""
        try:
            forecast_report = self.forecaster.generate_forecast_report(self.df, int(forecast_days))
            
            # Create forecast plot
            predictions_df = pd.DataFrame(forecast_report['predictions'])
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            
            fig = self.forecaster.plot_forecast(self.df, predictions_df, "Sales Forecast")
            
            # Save plot to file
            plot_file = "models/forecast_plot.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            # Create summary text
            summary = f"""
            **Forecast Summary:**
            - Model Used: {forecast_report['model_used'].title()}
            - R² Score: {forecast_report['model_metrics']['r2_test']:.4f}
            - RMSE: {forecast_report['model_metrics']['rmse_test']:.2f}
            - Total Predicted Sales: ${forecast_report['forecast_stats']['total_predicted_sales']:,.2f}
            - Mean Daily Prediction: ${forecast_report['forecast_stats']['mean_prediction']:,.2f}
            """
            
            return summary, plot_file
            
        except Exception as e:
            return f"Error generating forecast: {str(e)}", None
    
    def create_sales_charts(self):
        """Create sales visualization charts"""
        try:
            # Monthly sales trend
            self.df['date'] = pd.to_datetime(self.df['date'])
            monthly_sales = self.df.groupby(self.df['date'].dt.to_period('M'))['sales'].sum().reset_index()
            monthly_sales['date'] = monthly_sales['date'].astype(str)
            
            fig_monthly = px.line(monthly_sales, x='date', y='sales', 
                                 title="Monthly Sales Trend",
                                 labels={'sales': 'Total Sales ($)', 'date': 'Month'})
            
            # Product performance
            product_sales = self.df.groupby('product')['sales'].sum().sort_values(ascending=True)
            fig_products = px.bar(x=product_sales.values, y=product_sales.index, 
                                 orientation='h',
                                 title="Product Performance",
                                 labels={'x': 'Total Sales ($)', 'y': 'Product'})
            
            # Regional performance
            region_sales = self.df.groupby('region')['sales'].sum()
            fig_region = px.pie(values=region_sales.values, names=region_sales.index,
                               title="Sales by Region")
            
            return fig_monthly, fig_products, fig_region
            
        except Exception as e:
            return None, None, None
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            stats = self.rag_system.get_system_stats()
            if 'error' not in stats:
                return f"""
                **System Statistics:**
                - Total Documents: {stats['total_documents']}
                - LLM Model: {stats['model_used']}
                - Embedding Model: {stats['embedding_model']}
                - Max Context Length: {stats['max_context_length']}
                """
            else:
                return f"Error getting system stats: {stats['error']}"
        except Exception as e:
            return f"Error getting system stats: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Sales Forecasting AI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📊 Sales Forecasting AI")
            gr.Markdown("A comprehensive sales forecasting system with AI-powered insights and predictions.")
            
            with gr.Tabs():
                # Dashboard Tab
                with gr.Tab("🏠 Dashboard"):
                    gr.Markdown("## Sales Overview")
                    
                    # Key metrics
                    with gr.Row():
                        total_sales = gr.Number(value=self.df['sales'].sum(), label="Total Sales ($)")
                        avg_sales = gr.Number(value=self.df['sales'].mean(), label="Average Daily Sales ($)")
                        num_products = gr.Number(value=self.df['product'].nunique(), label="Total Products")
                        num_regions = gr.Number(value=self.df['region'].nunique(), label="Total Regions")
                    
                    # Charts
                    with gr.Row():
                        monthly_chart = gr.Plot(label="Monthly Sales Trend")
                        product_chart = gr.Plot(label="Product Performance")
                    
                    with gr.Row():
                        region_chart = gr.Plot(label="Sales by Region")
                    
                    # Update charts button
                    update_charts_btn = gr.Button("🔄 Update Charts")
                    update_charts_btn.click(
                        fn=self.create_sales_charts,
                        outputs=[monthly_chart, product_chart, region_chart]
                    )
                
                # AI Assistant Tab
                with gr.Tab("🤖 AI Assistant"):
                    gr.Markdown("## Ask AI About Your Sales Data")
                    gr.Markdown("""
                    Ask me anything about your sales data! Examples:
                    - "What were the top performing products in Q3?"
                    - "Show me sales trends for the Northeast region"
                    - "Predict sales for next quarter"
                    - "Which products have declining sales trends?"
                    """)
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Enter your question:",
                            placeholder="e.g., What were the top performing products in Q3 2023?",
                            lines=3
                        )
                        n_results = gr.Slider(
                            minimum=3, maximum=10, value=5, step=1,
                            label="Number of supporting chunks"
                        )
                    
                    ask_btn = gr.Button("🔍 Ask AI", variant="primary")
                    answer_output = gr.Textbox(label="AI Response", lines=10)
                    
                    ask_btn.click(
                        fn=self.answer_query,
                        inputs=[query_input, n_results],
                        outputs=answer_output
                    )
                
                # Forecasting Tab
                with gr.Tab("📈 Forecasting"):
                    gr.Markdown("## Generate Sales Forecasts")
                    gr.Markdown("Generate comprehensive sales forecasts using machine learning models.")
                    
                    with gr.Row():
                        forecast_days = gr.Slider(
                            minimum=30, maximum=365, value=90, step=30,
                            label="Forecast Period (days)"
                        )
                        forecast_btn = gr.Button("🚀 Generate Forecast", variant="primary")
                    
                    forecast_summary = gr.Markdown(label="Forecast Summary")
                    forecast_plot = gr.Image(label="Forecast Plot")
                    
                    forecast_btn.click(
                        fn=self.generate_forecast,
                        inputs=[forecast_days],
                        outputs=[forecast_summary, forecast_plot]
                    )
                
                # Data Analysis Tab
                with gr.Tab("📊 Data Analysis"):
                    gr.Markdown("## Data Analysis")
                    
                    # Data overview
                    gr.Markdown("### Data Overview")
                    data_preview = gr.Dataframe(
                        value=self.df.head(10),
                        label="First 10 rows of data"
                    )
                    
                    # Statistical summary
                    gr.Markdown("### Statistical Summary")
                    stats_df = self.df.describe()
                    stats_preview = gr.Dataframe(
                        value=stats_df,
                        label="Statistical Summary"
                    )
                    
                    # Correlation matrix
                    gr.Markdown("### Correlation Analysis")
                    numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        correlation_matrix = self.df[numeric_cols].corr()
                        corr_plot = gr.Plot(label="Correlation Matrix")
                        
                        def create_corr_plot():
                            fig = px.imshow(correlation_matrix, 
                                           title="Correlation Matrix",
                                           color_continuous_scale='RdBu')
                            return fig
                        
                        corr_plot.value = create_corr_plot()
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("## System Configuration")
                    
                    system_stats = gr.Markdown(label="System Information")
                    refresh_stats_btn = gr.Button("🔄 Refresh System Stats")
                    
                    refresh_stats_btn.click(
                        fn=self.get_system_stats,
                        outputs=system_stats
                    )
                    
                    # Initialize stats on load
                    system_stats.value = self.get_system_stats()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("Built with ❤️ using Gradio and AI")
        
        return demo

def main():
    """Main function to run the Gradio app"""
    app = GradioSalesForecasting()
    demo = app.create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main() 
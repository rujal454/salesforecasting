# Sales Forecasting AI with RAG (Retrieval-Augmented Generation)

A comprehensive sales forecasting system that combines historical data analysis, vector embeddings, and LLM-powered insights to provide intelligent sales predictions and analysis.

## 🚀 Features

### Phase 1: Data Ingestion & Processing
- ✅ Load historical sales data from CSV files
- ✅ Chunk data by time periods (monthly, quarterly, yearly)
- ✅ Convert various document formats (PDF, DOC) to plain text
- ✅ Automatic data preprocessing and cleaning
- ✅ Sample data generation for testing

### Phase 2: Embedding & Vector Storage
- ✅ Generate embeddings using OpenAI or SBERT
- ✅ Store in ChromaDB vector database with metadata
- ✅ Support for region, product, and date-based queries
- ✅ Efficient similarity search capabilities

### Phase 3: Retrieval-Augmented Answering
- ✅ Natural language query processing
- ✅ Top-k similar chunk retrieval
- ✅ LLM-powered analysis with context
- ✅ Multiple prompt templates for different query types

### Phase 4: User Interface
- ✅ Streamlit web application with modern UI
- ✅ Gradio alternative interface
- ✅ Interactive visualizations and insights
- ✅ Real-time query processing

## 📁 Project Structure

```
sales-forecasting/
├── data/                    # Historical sales data (CSV files)
├── src/                    # Core application code
│   ├── data_ingestion.py   # Phase 1: Data processing
│   ├── embeddings.py       # Phase 2: Vector embeddings
│   ├── retrieval.py        # Phase 3: RAG system
│   └── forecasting.py      # ML forecasting models
├── ui/                     # User interfaces
│   ├── streamlit_app.py    # Streamlit web app
│   └── gradio_app.py       # Gradio interface
├── models/                 # Trained models and vector stores
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── setup.py               # Automated setup script
├── run_pipeline.py        # Complete pipeline runner
└── README.md              # This file
```

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd sales-forecasting

# Run automated setup
python setup.py
```

### 2. Configure Environment (Optional)

If you want to use OpenAI for embeddings, edit the `.env` file:

```bash
# Copy example environment file
cp env_example.txt .env

# Edit .env file and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application

#### Option A: Streamlit UI (Recommended)
```bash
streamlit run ui/streamlit_app.py
```

#### Option B: Gradio UI
```bash
python ui/gradio_app.py
```

#### Option C: Complete Pipeline
```bash
python run_pipeline.py
```

## 📊 Usage Examples

### Natural Language Queries

Ask the AI assistant questions like:

- **Analysis**: "What were the top performing products in Q3 2023?"
- **Trends**: "Show me sales trends for the Northeast region"
- **Forecasting**: "Predict sales for next quarter based on historical data"
- **Insights**: "Which products have declining sales trends?"
- **Comparison**: "Compare sales performance between regions"

### Forecasting

Generate comprehensive sales forecasts with:

- **Time Series Analysis**: Automatic trend detection
- **Seasonal Patterns**: Identify recurring patterns
- **Feature Importance**: Understand key drivers
- **Multiple Models**: Linear regression and Random Forest
- **Performance Metrics**: R² score, RMSE, MAE

### Data Visualization

Interactive charts and insights:

- **Sales Trends**: Monthly, quarterly, and yearly views
- **Product Performance**: Top and bottom performers
- **Regional Analysis**: Geographic distribution
- **Correlation Analysis**: Feature relationships
- **Forecast Plots**: Historical vs predicted data

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./models/chroma_db

# Model Configuration
EMBEDDING_MODEL=sbert  # or "openai"
LLM_MODEL=gpt-3.5-turbo

# Application Configuration
MAX_CONTEXT_LENGTH=4000
DEFAULT_N_RESULTS=5
```

### Data Format

Place your CSV files in the `data/` directory with columns:

```csv
date,product,region,quantity,price,sales
2023-01-01,Laptop,North,2,1200.00,2400.00
2023-01-01,Phone,South,1,800.00,800.00
...
```

## 🧪 Testing

### Run Individual Phases

```bash
# Phase 1: Data Ingestion
python src/data_ingestion.py

# Phase 2: Embedding Generation
python src/embeddings.py

# Phase 3: Test RAG System
python src/retrieval.py

# Phase 4: Start UI
streamlit run ui/streamlit_app.py
```

### Jupyter Notebook

Explore the data interactively:

```bash
jupyter notebook notebooks/exploration.ipynb
```

## 📈 Advanced Features

### Custom Embedding Models

Switch between embedding models:

```python
# Use SBERT (default, free)
embedding_manager = EmbeddingManager(embedding_model="sbert")

# Use OpenAI (requires API key)
embedding_manager = EmbeddingManager(embedding_model="openai")
```

### Custom Forecasting Models

Extend the forecasting capabilities:

```python
from src.forecasting import SalesForecaster

forecaster = SalesForecaster()
forecaster.train_model(df, model_type='linear')  # or 'random_forest'
predictions = forecaster.predict_future(df, periods=90)
```

### Custom Prompt Templates

Add new query types in `src/retrieval.py`:

```python
templates = {
    "custom_analysis": """Your custom prompt template here..."""
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project directory
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **OpenAI API Errors**: Check your API key in `.env`
4. **Memory Issues**: Reduce `MAX_CONTEXT_LENGTH` in `.env`

### Logs

Check the logs for detailed error information:

```bash
tail -f pipeline.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- **OpenAI** for LLM capabilities
- **Streamlit** and **Gradio** for UI frameworks
- **Scikit-learn** for ML models

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub
4. Check the documentation

---

**Happy Forecasting! 📊🚀** 
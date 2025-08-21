# 🚀 Sales Forecasting AI

A comprehensive sales forecasting and analytics platform powered by AI, featuring Prophet time series forecasting and intelligent chatbot analysis.

## ✨ Features

### 📈 Advanced Sales Forecasting
- **Prophet Model Integration** - Professional time series forecasting with seasonality detection
- **Confidence Intervals** - Upper and lower bound predictions
- **Performance Metrics** - MAE, MAPE, and model accuracy indicators
- **Multiple Time Horizons** - Daily, weekly, monthly, quarterly, and yearly forecasts

### 🤖 AI-Powered Chatbot
- **OpenRouter Integration** - Powered by Mistral-7B-Instruct model
- **Context-Aware Analysis** - Understands your uploaded sales data
- **Natural Language Queries** - Ask questions in plain English
- **Business Intelligence** - Actionable insights and recommendations

### 📊 Comprehensive Analytics
- **Top Product Analysis** - Identify best-performing products
- **Customer Insights** - Analyze customer behavior and spending patterns
- **Seasonal Patterns** - Monthly, quarterly, and weekly trend analysis
- **Performance Metrics** - Complete KPI dashboard
- **Decline Analysis** - Identify underperforming segments

### 🔧 Technical Features
- **Multi-Encoding Support** - Handles various CSV file encodings
- **Real-time Processing** - FastAPI backend with auto-reload
- **Interactive UI** - Streamlit frontend with modern design
- **Data Visualization** - Charts and graphs for trend analysis

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.13
- **Frontend**: Streamlit
- **Forecasting**: Prophet (Facebook's time series forecasting tool)
- **AI**: OpenRouter API with Mistral-7B-Instruct
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit charts

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenRouter API key (get from [openrouter.ai](https://openrouter.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd salesforecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   pip install prophet python-dotenv requests
   ```

3. **Configure environment**
   ```bash
   cp app/.env.example app/.env
   # Edit app/.env and add your OpenRouter API key
   ```

4. **Start the backend**
   ```bash
   cd app
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Start the frontend**
   ```bash
   streamlit run app/frontend/ui/streamlit_app.py
   ```

6. **Open your browser**
   - Frontend: http://localhost:8501 (or 8502, 8503)
   - Backend API: http://localhost:8000

## 📝 Usage

### 1. Upload Sales Data
- Upload a CSV file with sales data
- Required columns: date column and sales/revenue column
- Supports various encodings (UTF-8, CP1252, Latin-1, etc.)

### 2. Generate Forecasts
- Click "Forecast" to generate Prophet-powered predictions
- View confidence intervals and model performance metrics
- Analyze trends and seasonal patterns

### 3. Ask the AI Chatbot
Try these example queries:
- "What are the top sales products?"
- "Forecast sales for next month"
- "Analyze seasonal patterns"
- "Show performance metrics"
- "Which products are declining?"

## 🔧 Configuration

### Environment Variables (app/.env)
```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=your_api_key_here
MISTRAL_MODEL=mistralai/mistral-7b-instruct

# Optional configurations
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=.chroma
```

## 📊 Supported Data Formats

Your CSV should include:
- **Date column**: Any column with 'date', 'time' in the name
- **Sales column**: Any column with 'sales', 'revenue', 'amount', 'total'
- **Product column** (optional): For product analysis
- **Customer column** (optional): For customer analysis

Example CSV structure:
```csv
ORDERDATE,SALES,PRODUCTLINE,CUSTOMERNAME
2023-01-01,1500.00,Classic Cars,ABC Corp
2023-01-02,2300.50,Motorcycles,XYZ Ltd
```

## 🎯 API Endpoints

- `POST /forecast` - Generate sales forecasts
- `POST /chat` - AI chatbot queries
- `POST /embed-index` - Index data for embeddings
- `GET /test-openrouter` - Test API connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the logs in the terminal
2. Ensure your OpenRouter API key is valid
3. Verify your CSV file format
4. Check that all dependencies are installed

## 🔮 Future Enhancements

- [ ] Multiple forecasting models (ARIMA, LSTM)
- [ ] Advanced data visualization
- [ ] Export functionality
- [ ] User authentication
- [ ] Database integration
- [ ] Real-time data streaming

# Sales Forecasting AI - Flask + Streamlit Architecture

A modern sales forecasting application with **Flask backend API** and **Streamlit frontend**, powered by **Mistral AI** for intelligent question answering.

## 🏗️ Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Streamlit     │ ◄────────────► │   Flask Backend │
│   Frontend      │                │                 │
│   (Port 8501)   │                │ (Port 5000)     │
└─────────────────┘                └─────────────────┘
                                           │
                                           ▼
                                    ┌─────────────────┐
                                    │   Mistral AI    │
                                    │   (Local/HF)    │
                                    └─────────────────┘
```

## ✨ Features

- **📁 CSV Upload**: Upload sales data via Streamlit interface
- **🔮 Sales Forecasting**: Time-series forecasting using scikit-learn
- **🤖 AI Assistant**: Mistral AI-powered question answering
- **📊 Interactive Charts**: Plotly-based data visualization
- **🔄 Session Management**: Multi-user session handling
- **📈 Real-time Analysis**: Live data processing and insights

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Application

**Option A: Use the startup script (Recommended)**
```bash
python start_app.py
```

**Option B: Start manually**
```bash
# Terminal 1: Start Flask backend
python app.py

# Terminal 2: Start Streamlit frontend
streamlit run streamlit_frontend.py
```

### 3. Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:5000

## 📋 Usage Guide

### 1. Upload Data
1. Open http://localhost:8501
2. Navigate to "Upload Data" page
3. Upload your CSV file with sales data
4. Preview the data and click "Upload to Backend"

### 2. Generate Forecasts
1. Go to "Dashboard" page
2. View your sales data visualization
3. Adjust forecast periods (7-90 days)
4. Click "Generate Forecast"
5. View forecast charts and metrics

### 3. Ask AI Questions
1. Navigate to "AI Assistant" page
2. Ask questions about your sales data
3. Get AI-powered insights using Mistral
4. View context information and analysis

## 📊 Data Format

Your CSV file should contain columns like:

```csv
date,product,region,quantity,price,sales
2023-01-01,Product A,North,10,100.00,1000.00
2023-01-02,Product B,South,5,200.00,1000.00
...
```

**Required columns:**
- `date` (or similar date column)
- `sales` (or similar sales/revenue column)

**Optional columns:**
- `product`, `region`, `quantity`, `price`, etc.

## 🔧 API Endpoints

### Backend API (Flask)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload CSV file |
| `/forecast` | POST | Generate sales forecast |
| `/ask` | POST | Ask AI question |
| `/session/<id>` | GET | Get session data |
| `/sessions` | GET | List all sessions |
| `/delete/<id>` | DELETE | Delete session |

### Example API Usage

```python
import requests

# Upload file
with open('sales_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/upload', files={'file': f})
    session_id = response.json()['session_id']

# Generate forecast
forecast_data = {
    'session_id': session_id,
    'periods': 30
}
response = requests.post('http://localhost:5000/forecast', json=forecast_data)

# Ask question
question_data = {
    'session_id': session_id,
    'question': 'What are the top performing products?'
}
response = requests.post('http://localhost:5000/ask', json=question_data)
```

## 🤖 Mistral AI Integration

The application uses **Mistral-7B-Instruct-v0.2** for intelligent question answering:

### Features
- **Local Inference**: Runs locally using Hugging Face Transformers
- **8-bit Quantization**: Memory-efficient model loading
- **Fallback Responses**: Template-based responses if model unavailable
- **Context-Aware**: Analyzes sales data and forecasts

### Installation
```bash
# Install Mistral dependencies
pip install transformers torch accelerate bitsandbytes

# Optional: For GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Model Configuration
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Quantization**: 8-bit for memory efficiency
- **Max Tokens**: 512
- **Temperature**: 0.7

## 📁 Project Structure

```
sales-forecasting/
├── app.py                    # Flask backend API
├── streamlit_frontend.py     # Streamlit frontend
├── start_app.py             # Startup script
├── requirements.txt         # Python dependencies
├── README_FLASK_STREAMLIT.md # This file
├── uploads/                 # Uploaded CSV files
├── data/                    # Sample data
├── src/                     # Legacy components
└── ui/                      # Legacy UI components
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configuration:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Mistral Configuration
MISTRAL_MODEL=mistralai/Mistral-7B-Instruct-v0.2
MISTRAL_MAX_TOKENS=512
MISTRAL_TEMPERATURE=0.7

# Upload Configuration
MAX_FILE_SIZE=16777216  # 16MB
UPLOAD_FOLDER=uploads
```

### Customization

#### Change Backend Port
```python
# In app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

#### Change Frontend Port
```bash
streamlit run streamlit_frontend.py --server.port 8502
```

#### Use Different Mistral Model
```python
# In app.py, MistralAI class
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Change this
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start application
python start_app.py
```

### Production Deployment

#### Backend (Flask)
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t sales-forecast-backend .
docker run -p 5000:5000 sales-forecast-backend
```

#### Frontend (Streamlit)
```bash
# Deploy to Streamlit Cloud
# 1. Push code to GitHub
# 2. Connect repository to Streamlit Cloud
# 3. Deploy automatically

# Or run locally
streamlit run streamlit_frontend.py --server.port 8501
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Backend Connection Failed
```bash
# Check if Flask is running
curl http://localhost:5000/health

# Check logs
python app.py
```

#### 2. Mistral Model Not Loading
```bash
# Install dependencies
pip install transformers torch accelerate bitsandbytes

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Memory Issues
```python
# Reduce model precision in app.py
load_in_8bit=True  # Already enabled
load_in_4bit=True  # For even more memory savings
```

#### 4. Port Already in Use
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

### Performance Optimization

#### For Large Datasets
```python
# In app.py, SalesForecaster class
# Increase chunk size for processing
CHUNK_SIZE = 10000
```

#### For Better AI Responses
```python
# In app.py, MistralAI class
# Increase max tokens for longer responses
max_new_tokens=1024
```

## 📈 Example Workflow

1. **Upload Sales Data**
   ```
   CSV File → Streamlit Upload → Flask Backend → Session Storage
   ```

2. **Generate Forecast**
   ```
   User Request → Flask API → Scikit-learn Model → Forecast Data → Charts
   ```

3. **Ask AI Questions**
   ```
   Question → Flask API → Mistral AI → Context Analysis → Response
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Mistral AI** for the language model
- **Streamlit** for the frontend framework
- **Flask** for the backend API
- **Plotly** for interactive visualizations
- **Scikit-learn** for forecasting models

---

**Happy Forecasting! 📊🚀**

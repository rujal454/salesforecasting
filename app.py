#!/usr/bin/env python3
"""
Flask Backend for Sales Forecasting AI
Handles CSV uploads, forecasting, and Mistral-powered question answering
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pickle
import uuid
import re

# ML and forecasting imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import torch

# Optional encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# Mistral model imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads directory
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Global variables for session management
sessions = {}  # Store session data: {session_id: {data, forecast, model}}


def detect_file_encoding(file_path: str, default_encoding: str = 'utf-8') -> str:
    """Detect file encoding using chardet, fallback to default."""
    if HAS_CHARDET:
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(200_000)
            result = chardet.detect(sample) if sample else {}
            encoding = result.get('encoding') or default_encoding
            logger.info(f"Detected encoding for {file_path}: {encoding}")
            return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return default_encoding
    return default_encoding

class SalesForecaster:
    """Sales forecasting using time series analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_data(self, df):
        """Prepare data for forecasting"""
        # Ensure date column exists
        if 'date' not in df.columns:
            # Try to find date-like column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df['date'] = pd.to_datetime(df[date_cols[0]])
            else:
                # Create synthetic dates if none exist
                df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure sales column exists
        if 'sales' not in df.columns:
            # Try to find sales-like column
            sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower()]
            if sales_cols:
                df['sales'] = df[sales_cols[0]]
            else:
                # Use first numeric column as sales
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['sales'] = df[numeric_cols[0]]
                else:
                    raise ValueError("No sales column found in data")
        
        # Aggregate by date
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('date')
        
        # Create features
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['quarter'] = daily_sales['date'].dt.quarter
        daily_sales['year'] = daily_sales['date'].dt.year
        
        # Create lag features
        daily_sales['sales_lag1'] = daily_sales['sales'].shift(1)
        daily_sales['sales_lag7'] = daily_sales['sales'].shift(7)
        daily_sales['sales_lag30'] = daily_sales['sales'].shift(30)
        
        # Rolling averages
        daily_sales['sales_ma7'] = daily_sales['sales'].rolling(window=7).mean()
        daily_sales['sales_ma30'] = daily_sales['sales'].rolling(window=30).mean()
        
        # Remove NaN values
        daily_sales = daily_sales.dropna()
        
        return daily_sales
    
    def train(self, df):
        """Train the forecasting model"""
        # Prepare data
        daily_sales = self.prepare_data(df)
        
        # Features for training
        feature_cols = ['day_of_week', 'month', 'quarter', 'year', 
                       'sales_lag1', 'sales_lag7', 'sales_lag30',
                       'sales_ma7', 'sales_ma30']
        
        X = daily_sales[feature_cols]
        y = daily_sales['sales']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': self.model.score(X_scaled, y),
            'training_samples': len(X)
        }
    
    def forecast(self, df, periods=30):
        """Generate forecast for next N periods"""
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Prepare data
        daily_sales = self.prepare_data(df)
        
        # Get last date
        last_date = daily_sales['date'].max()
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=periods, freq='D')
        
        # Create future dataframe
        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['quarter'] = future_df['date'].dt.quarter
        future_df['year'] = future_df['date'].dt.year
        
        # Get last values for lag features
        last_sales = daily_sales['sales'].iloc[-1]
        last_sales_lag1 = daily_sales['sales'].iloc[-1] if len(daily_sales) > 0 else last_sales
        last_sales_lag7 = daily_sales['sales'].iloc[-7] if len(daily_sales) >= 7 else last_sales
        last_sales_lag30 = daily_sales['sales'].iloc[-30] if len(daily_sales) >= 30 else last_sales
        last_ma7 = daily_sales['sales'].rolling(window=7).mean().iloc[-1]
        last_ma30 = daily_sales['sales'].rolling(window=30).mean().iloc[-1]
        
        # Initialize lag features
        future_df['sales_lag1'] = last_sales_lag1
        future_df['sales_lag7'] = last_sales_lag7
        future_df['sales_lag30'] = last_sales_lag30
        future_df['sales_ma7'] = last_ma7
        future_df['sales_ma30'] = last_ma30
        
        # Feature columns
        feature_cols = ['day_of_week', 'month', 'quarter', 'year', 
                       'sales_lag1', 'sales_lag7', 'sales_lag30',
                       'sales_ma7', 'sales_ma30']
        
        # Generate predictions
        predictions = []
        for i in range(len(future_df)):
            # Prepare features for this prediction
            X_pred = future_df[feature_cols].iloc[i:i+1]
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predict
            pred = self.model.predict(X_pred_scaled)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
            
            # Update lag features for next prediction
            if i < len(future_df) - 1:
                future_df.loc[future_df.index[i+1], 'sales_lag1'] = pred
                if i >= 6:
                    future_df.loc[future_df.index[i+1], 'sales_lag7'] = predictions[i-6]
                if i >= 29:
                    future_df.loc[future_df.index[i+1], 'sales_lag30'] = predictions[i-29]
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions
        })
        
        return forecast_df

class MistralAI:
    """Mistral AI integration for question answering"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Mistral model"""
        if not MISTRAL_AVAILABLE:
            logger.warning("Mistral model not available. Install transformers and torch.")
            return
        
        try:
            # Allow model override via env var; default stays light
            model_name = os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
            max_new_tokens = int(os.getenv("MISTRAL_MAX_TOKENS", "512"))
            temperature = float(os.getenv("MISTRAL_TEMPERATURE", "0.2"))
            do_sample = os.getenv("MISTRAL_DO_SAMPLE", "false").lower() == "true"
            quant_mode = os.getenv("MISTRAL_QUANT", "auto").lower()  # auto|4bit|8bit|none
            
            logger.info(f"Loading model: {model_name} (quant={quant_mode})")
            
            # Device selection
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            device_map = "auto" if (has_cuda or has_mps) else "cpu"
            load_kwargs = {}
            
            # Quantization (GPU only). On CPU, skip quantization flags
            if device_map != "cpu":
                if quant_mode in ("4bit", "auto") and has_cuda:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                elif quant_mode in ("8bit", "auto") and has_cuda:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    load_kwargs["torch_dtype"] = torch.bfloat16 if has_cuda else torch.float32
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            # Load tokenizer/model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                **load_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            logger.info("Falling back to template-based responses")
    
    def generate_response(self, question, context_data):
        """Generate response using Mistral model"""
        if self.pipeline is None:
            return self._template_response(question, context_data)
        
        try:
            # Create prompt
            prompt = self._create_prompt(question, context_data)
            
            # Generate response
            response = self.pipeline(prompt)
            generated_text = response[0]['generated_text']
            
            # Extract the response text after the last [/INST]
            split_token = '[/INST]'
            if split_token in generated_text:
                response_text = generated_text.split(split_token)[-1].strip()
            else:
                # Fallback: remove the prompt prefix if present
                response_text = generated_text.replace(prompt, "").strip()
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating Mistral response: {e}")
            return self._template_response(question, context_data)
    
    def _create_prompt(self, question, context_data):
        """Create prompt for Mistral model"""
        prompt = f"""<s>[INST] You are an expert sales forecasting AI assistant. Analyze the following sales data and answer the user's question.

Context Data:
{context_data}

Question: {question}

Please provide a comprehensive, data-driven answer based on the context provided. Include specific numbers, trends, and insights where available. If the question asks for top products, regions, or specific metrics, provide the exact data from the context. Be specific and quantitative in your response. [/INST]"""
        
        return prompt
    
    def _template_response(self, question, context_data):
        """Fallback template-based response"""
        question_lower = question.lower()
        
        if 'forecast' in question_lower or 'predict' in question_lower:
            return "Based on the historical data, I can see trends that suggest future sales patterns. The forecasting model indicates potential growth areas and seasonal variations."
        elif 'trend' in question_lower:
            return "The sales data shows clear trends over time. There are seasonal patterns and growth indicators that suggest continued performance in key areas."
        elif 'top' in question_lower or 'best' in question_lower:
            return "The data indicates strong performance in several key areas. The top performers show consistent growth and market leadership."
        else:
            return "Based on the sales data provided, I can offer insights about performance, trends, and opportunities. The data shows various patterns that can inform business decisions."

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session_id():
    """Create unique session ID"""
    return str(uuid.uuid4())

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mistral_available': MISTRAL_AVAILABLE
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload CSV file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Create session ID
            session_id = create_session_id()
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(filepath)
            
            # Read and validate CSV with robust encoding handling
            try:
                # Detect encoding and attempt read
                encodings_to_try = []
                detected = detect_file_encoding(filepath, default_encoding='utf-8')
                encodings_to_try.append(detected)
                # Add common fallbacks
                for enc in ['utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']:
                    if enc not in encodings_to_try:
                        encodings_to_try.append(enc)
                
                last_error = None
                df = None
                for enc in encodings_to_try:
                    try:
                        df = pd.read_csv(filepath, encoding=enc)
                        logger.info(f"Successfully read CSV using encoding: {enc}")
                        break
                    except UnicodeDecodeError as ue:
                        last_error = ue
                        logger.warning(f"UnicodeDecodeError with encoding {enc}: {ue}")
                        continue
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Read error with encoding {enc}: {e}")
                        continue
                
                if df is None:
                    raise last_error or Exception("Failed to read CSV with available encodings")
                
                if len(df) == 0:
                    return jsonify({'error': 'CSV file is empty'}), 400
                
                # Store session data
                sessions[session_id] = {
                    'filepath': filepath,
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'shape': df.shape,
                    'upload_time': datetime.now().isoformat()
                }
                
                return jsonify({
                    'session_id': session_id,
                    'filename': filename,
                    'rows': len(df),
                    'columns': df.columns.tolist(),
                    'message': 'File uploaded successfully'
                })
                
            except Exception as e:
                logger.error(f"Error reading CSV: {e}")
                return jsonify({'error': f"Error reading CSV. Try saving as UTF-8 or use a different encoding (e.g., CP1252). Details: {str(e)}"}), 400
        
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/forecast', methods=['POST'])
def generate_forecast():
    """Generate sales forecast endpoint"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        periods = data.get('periods', 30)
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session_data = sessions[session_id]
        df = pd.DataFrame(session_data['data'])
        
        # Initialize forecaster
        forecaster = SalesForecaster()
        
        # Train model
        training_metrics = forecaster.train(df)
        
        # Generate forecast
        forecast_df = forecaster.forecast(df, periods=periods)
        
        # Store forecast in session
        sessions[session_id]['forecast'] = forecast_df.to_dict('records')
        sessions[session_id]['forecaster'] = forecaster
        sessions[session_id]['training_metrics'] = training_metrics
        
        return jsonify({
            'session_id': session_id,
            'forecast': forecast_df.to_dict('records'),
            'training_metrics': training_metrics,
            'message': 'Forecast generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({'error': f'Forecast failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask question about sales data endpoint"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question')
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        session_data = sessions[session_id]
        
        # Try direct, rule-based answer first for precision and speed
        direct = _direct_answer_from_data(question, session_data)
        if direct:
            return jsonify({
                'session_id': session_id,
                'question': question,
                'answer': direct,
                'context_summary': {
                    'data_rows': session_data['shape'][0],
                    'data_columns': session_data['shape'][1],
                    'has_forecast': 'forecast' in session_data
                },
                'source': 'direct'
            })

        # Prepare context data for LLM
        context_data = _prepare_context(session_data)
        
        # Initialize Mistral AI
        mistral_ai = MistralAI()
        
        # Generate response
        response = mistral_ai.generate_response(question, context_data)
        
        return jsonify({
            'session_id': session_id,
            'question': question,
            'answer': response,
            'context_summary': {
                'data_rows': session_data['shape'][0],
                'data_columns': session_data['shape'][1],
                'has_forecast': 'forecast' in session_data
            },
            'source': 'llm'
        })
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        return jsonify({'error': f'Question answering failed: {str(e)}'}), 500

def _prepare_context(session_data):
    """Prepare context data for Mistral model"""
    df = pd.DataFrame(session_data['data'])
    
    context_parts = []
    
    # Basic data info
    context_parts.append(f"Dataset: {session_data['shape'][0]} rows, {session_data['shape'][1]} columns")
    context_parts.append(f"Columns: {', '.join(session_data['columns'])}")
    
    # Data summary
    if 'sales' in df.columns:
        context_parts.append(f"Total sales: ${df['sales'].sum():,.2f}")
        context_parts.append(f"Average sales: ${df['sales'].mean():,.2f}")
        context_parts.append(f"Sales range: ${df['sales'].min():,.2f} to ${df['sales'].max():,.2f}")
        
        # Top products analysis
        if 'product' in df.columns:
            top_products = df.groupby('product')['sales'].sum().sort_values(ascending=False).head(10)
            context_parts.append("Top 10 Products by Sales:")
            for i, (product, sales) in enumerate(top_products.items(), 1):
                context_parts.append(f"  {i}. {product}: ${sales:,.2f}")
        
        # Regional analysis
        if 'region' in df.columns:
            top_regions = df.groupby('region')['sales'].sum().sort_values(ascending=False)
            context_parts.append("Sales by Region:")
            for region, sales in top_regions.items():
                context_parts.append(f"  {region}: ${sales:,.2f}")
    
    # Date range and trends if available
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        try:
            df['date'] = pd.to_datetime(df[date_cols[0]])
            context_parts.append(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            # Monthly trends
            if 'sales' in df.columns:
                monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()
                context_parts.append("Monthly Sales Trends:")
                for month, sales in monthly_sales.tail(6).items():  # Last 6 months
                    context_parts.append(f"  {month}: ${sales:,.2f}")
        except Exception as e:
            context_parts.append(f"Date processing error: {e}")
    
    # Sample data points for context
    if 'sales' in df.columns:
        context_parts.append("Sample Data Points:")
        sample_data = df.head(5)
        for _, row in sample_data.iterrows():
            if 'product' in df.columns and 'sales' in df.columns:
                context_parts.append(f"  {row.get('product', 'Unknown')}: ${row['sales']:,.2f}")
            elif 'sales' in df.columns:
                context_parts.append(f"  Sales: ${row['sales']:,.2f}")
    
    # Forecast info if available
    if 'forecast' in session_data:
        forecast_df = pd.DataFrame(session_data['forecast'])
        context_parts.append(f"Forecast generated for {len(forecast_df)} periods")
        if 'forecast' in forecast_df.columns:
            context_parts.append(f"Forecast range: ${forecast_df['forecast'].min():,.2f} to ${forecast_df['forecast'].max():,.2f}")
            context_parts.append(f"Average forecast: ${forecast_df['forecast'].mean():,.2f}")
    
    # Training metrics if available
    if 'training_metrics' in session_data:
        metrics = session_data['training_metrics']
        context_parts.append(f"Model performance - MAE: ${metrics['mae']:,.2f}, R²: {metrics['r2']:.3f}")
    
    return "\n".join(context_parts)

def _direct_answer_from_data(question: str, session_data: dict) -> str | None:
    """Attempt to answer simple, frequent queries directly from the data without LLM.
    Supports: top N products, top N regions, basic totals.
    Returns a string if handled, else None.
    """
    try:
        df = pd.DataFrame(session_data['data'])
        if df.empty:
            return None

        q = question.lower()
        # Detect N for "top N" queries; default to 5
        match = re.search(r"top\s*(\d+)", q)
        top_n = int(match.group(1)) if match else 5
        top_n = max(1, min(top_n, 50))

        # Identify sales column
        sales_col = None
        for cand in ['sales', 'revenue', 'amount']:
            if cand in df.columns:
                sales_col = cand
                break
        if sales_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sales_col = numeric_cols[0]

        # Top products
        if 'product' in df.columns and sales_col and ('product' in q or 'item' in q):
            top = df.groupby('product')[sales_col].sum().sort_values(ascending=False).head(top_n)
            lines = [f"Top {len(top)} products by {sales_col}:"]
            for i, (name, val) in enumerate(top.items(), 1):
                lines.append(f"{i}. {name}: ${float(val):,.2f}")
            return "\n".join(lines)

        # Top regions
        if 'region' in df.columns and sales_col and ('region' in q or 'country' in q or 'state' in q or 'area' in q):
            top = df.groupby('region')[sales_col].sum().sort_values(ascending=False).head(top_n)
            lines = [f"Top {len(top)} regions by {sales_col}:"]
            for i, (name, val) in enumerate(top.items(), 1):
                lines.append(f"{i}. {name}: ${float(val):,.2f}")
            return "\n".join(lines)

        # Total/average sales quick answers
        if sales_col and ('total' in q or 'average' in q or 'avg' in q):
            total_val = float(df[sales_col].sum())
            avg_val = float(df[sales_col].mean())
            return f"Total {sales_col}: ${total_val:,.2f}. Average per row: ${avg_val:,.2f}."

        return None
    except Exception:
        return None

@app.route('/session/<session_id>', methods=['GET'])
def get_session_data(session_id):
    """Get session data endpoint"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = sessions[session_id]
    
    return jsonify({
        'session_id': session_id,
        'filename': os.path.basename(session_data['filepath']),
        'upload_time': session_data['upload_time'],
        'data_shape': session_data['shape'],
        'has_forecast': 'forecast' in session_data,
        'has_training_metrics': 'training_metrics' in session_data
    })

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append({
            'session_id': session_id,
            'filename': os.path.basename(session_data['filepath']),
            'upload_time': session_data['upload_time'],
            'data_shape': session_data['shape'],
            'has_forecast': 'forecast' in session_data
        })
    
    return jsonify({'sessions': session_list})

@app.route('/delete/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete session and associated files"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        # Delete file
        filepath = sessions[session_id]['filepath']
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Remove session data
        del sessions[session_id]
        
        return jsonify({'message': 'Session deleted successfully'})
        
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return jsonify({'error': 'Failed to delete session'}), 500

if __name__ == '__main__':
    print("🚀 Starting Sales Forecasting AI Flask Backend...")
    print("📊 API Endpoints:")
    print("  POST /upload - Upload CSV file")
    print("  POST /forecast - Generate sales forecast")
    print("  POST /ask - Ask questions about data")
    print("  GET /health - Health check")
    print("  GET /sessions - List active sessions")
    print("  GET /session/<id> - Get session data")
    print("  DELETE /delete/<id> - Delete session")
    print("\n🔗 Frontend URL: http://localhost:8501")
    print("🔗 Backend URL: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

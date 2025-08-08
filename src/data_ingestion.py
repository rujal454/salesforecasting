"""
Phase 1: Data Ingestion & Processing
Load historical sales CSVs, chunk by time period, convert documents to text
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import PyPDF2
from docx import Document
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.processed_data = []
        
    def load_csv_files(self) -> List[pd.DataFrame]:
        """Load all CSV files from the data directory"""
        csv_files = list(self.data_dir.glob("*.csv"))
        dataframes = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {csv_file.name} with {len(df)} rows")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                
        return dataframes
    
    def chunk_by_time_period(self, df: pd.DataFrame, 
                           date_column: str = 'date',
                           period: str = 'monthly') -> List[Dict[str, Any]]:
        """
        Chunk data by time period (monthly, quarterly, yearly)
        
        Args:
            df: DataFrame with sales data
            date_column: Name of the date column
            period: 'monthly', 'quarterly', or 'yearly'
        """
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        chunks = []
        
        if period == 'monthly':
            grouped = df.groupby(df[date_column].dt.to_period('M'))
        elif period == 'quarterly':
            grouped = df.groupby(df[date_column].dt.to_period('Q'))
        elif period == 'yearly':
            grouped = df.groupby(df[date_column].dt.to_period('Y'))
        else:
            raise ValueError("Period must be 'monthly', 'quarterly', or 'yearly'")
        
        for period_name, group in grouped:
            chunk_data = {
                'period': str(period_name),
                'period_type': period,
                'start_date': group[date_column].min(),
                'end_date': group[date_column].max(),
                'total_sales': group.get('sales', 0).sum() if 'sales' in group.columns else 0,
                'total_quantity': group.get('quantity', 0).sum() if 'quantity' in group.columns else 0,
                'unique_products': group.get('product', []).nunique() if 'product' in group.columns else 0,
                'unique_regions': group.get('region', []).nunique() if 'region' in group.columns else 0,
                'data': group.to_dict('records'),
                'summary_text': self._generate_summary_text(group, period_name)
            }
            chunks.append(chunk_data)
            
        return chunks
    
    def _generate_summary_text(self, group: pd.DataFrame, period_name: str) -> str:
        """Generate a text summary of the chunk data"""
        summary_parts = [f"Sales data for {period_name}:"]
        
        if 'sales' in group.columns:
            total_sales = group['sales'].sum()
            avg_sales = group['sales'].mean()
            summary_parts.append(f"Total sales: ${total_sales:,.2f}")
            summary_parts.append(f"Average sales: ${avg_sales:,.2f}")
        
        if 'product' in group.columns:
            top_products = group.groupby('product')['sales'].sum().nlargest(3)
            summary_parts.append("Top products by sales:")
            for product, sales in top_products.items():
                summary_parts.append(f"  - {product}: ${sales:,.2f}")
        
        if 'region' in group.columns:
            top_regions = group.groupby('region')['sales'].sum().nlargest(3)
            summary_parts.append("Top regions by sales:")
            for region, sales in top_regions.items():
                summary_parts.append(f"  - {region}: ${sales:,.2f}")
        
        return " ".join(summary_parts)
    
    def process_documents(self, docs_dir: str = "data/documents") -> List[str]:
        """Convert PDF and DOC files to plain text"""
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            logger.warning(f"Documents directory {docs_dir} does not exist")
            return []
        
        texts = []
        
        # Process PDF files
        for pdf_file in docs_path.glob("*.pdf"):
            try:
                text = self._extract_pdf_text(pdf_file)
                texts.append({
                    'source': str(pdf_file),
                    'text': text,
                    'type': 'pdf'
                })
                logger.info(f"Processed PDF: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {e}")
        
        # Process DOC files
        for doc_file in docs_path.glob("*.docx"):
            try:
                text = self._extract_doc_text(doc_file)
                texts.append({
                    'source': str(doc_file),
                    'text': text,
                    'type': 'docx'
                })
                logger.info(f"Processed DOC: {doc_file.name}")
            except Exception as e:
                logger.error(f"Error processing DOC {doc_file}: {e}")
        
        return texts
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_doc_text(self, doc_path: Path) -> str:
        """Extract text from DOCX file"""
        doc = Document(doc_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def create_sample_data(self):
        """Create sample sales data for testing"""
        np.random.seed(42)
        
        # Generate sample data
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        products = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        data = []
        for date in dates:
            for _ in range(np.random.randint(5, 15)):  # 5-15 sales per day
                product = np.random.choice(products)
                region = np.random.choice(regions)
                quantity = np.random.randint(1, 5)
                price = np.random.uniform(100, 2000)
                sales = quantity * price
                
                data.append({
                    'date': date,
                    'product': product,
                    'region': region,
                    'quantity': quantity,
                    'price': price,
                    'sales': sales
                })
        
        df = pd.DataFrame(data)
        sample_file = self.data_dir / "sample_sales_data.csv"
        df.to_csv(sample_file, index=False)
        logger.info(f"Created sample data: {sample_file}")
        return df
    
    def process_all_data(self, create_sample: bool = True) -> Dict[str, Any]:
        """Main method to process all data"""
        if create_sample:
            self.create_sample_data()
        
        # Load CSV files
        dataframes = self.load_csv_files()
        
        all_chunks = []
        all_documents = []
        
        # Process each dataframe
        for i, df in enumerate(dataframes):
            logger.info(f"Processing dataframe {i+1}/{len(dataframes)}")
            
            # Chunk by different time periods
            for period in ['monthly', 'quarterly']:
                chunks = self.chunk_by_time_period(df, period=period)
                all_chunks.extend(chunks)
        
        # Process documents
        all_documents = self.process_documents()
        
        # Save processed data
        processed_data = {
            'chunks': all_chunks,
            'documents': all_documents,
            'metadata': {
                'total_chunks': len(all_chunks),
                'total_documents': len(all_documents),
                'processed_at': datetime.now().isoformat()
            }
        }
        
        # Save to JSON for next phase
        output_file = Path("models/processed_data.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, default=str, indent=2)
        
        logger.info(f"Processed data saved to {output_file}")
        logger.info(f"Total chunks: {len(all_chunks)}")
        logger.info(f"Total documents: {len(all_documents)}")
        
        return processed_data

if __name__ == "__main__":
    # Run data ingestion
    ingestion = DataIngestion()
    processed_data = ingestion.process_all_data()
    print("Data ingestion completed successfully!") 
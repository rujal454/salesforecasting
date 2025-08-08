"""
Phase 2: Embedding & Vector Storage
Use OpenAI or SBERT to embed chunks, store in Chroma with metadata
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, 
                 embedding_model: str = "sbert",  # "openai" or "sbert"
                 chroma_persist_dir: str = "models/chroma_db"):
        self.embedding_model = embedding_model
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding models
        if embedding_model == "openai":
            self._init_openai()
        elif embedding_model == "sbert":
            self._init_sbert()
        else:
            raise ValueError("embedding_model must be 'openai' or 'sbert'")
        
        # Initialize ChromaDB
        self._init_chroma()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        logger.info("Initialized OpenAI client")
    
    def _init_sbert(self):
        """Initialize SBERT model"""
        try:
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized SBERT model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Could not load SBERT model: {e}")
            # Fallback to a simpler model
            self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logger.info("Using fallback SBERT model: paraphrase-MiniLM-L3-v2")
    
    def _init_chroma(self):
        """Initialize ChromaDB client"""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="sales_forecasting",
            metadata={"description": "Sales forecasting embeddings"}
        )
        logger.info(f"Initialized ChromaDB at {self.chroma_persist_dir}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        if self.embedding_model == "openai":
            return self._generate_openai_embedding(text)
        else:
            return self._generate_sbert_embedding(text)
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    def _generate_sbert_embedding(self, text: str) -> List[float]:
        """Generate embedding using SBERT"""
        try:
            embedding = self.sbert_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating SBERT embedding: {e}")
            raise
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks and generate embeddings"""
        processed_chunks = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
            try:
                # Create text representation for embedding
                text_content = self._create_chunk_text(chunk)
                
                # Generate embedding
                embedding = self.generate_embedding(text_content)
                
                # Create metadata
                metadata = {
                    'period': chunk.get('period', ''),
                    'period_type': chunk.get('period_type', ''),
                    'start_date': str(chunk.get('start_date', '')),
                    'end_date': str(chunk.get('end_date', '')),
                    'total_sales': chunk.get('total_sales', 0),
                    'total_quantity': chunk.get('total_quantity', 0),
                    'unique_products': chunk.get('unique_products', 0),
                    'unique_regions': chunk.get('unique_regions', 0),
                    'chunk_type': 'sales_data'
                }
                
                processed_chunk = {
                    'id': f"chunk_{i}",
                    'text': text_content,
                    'embedding': embedding,
                    'metadata': metadata,
                    'original_data': chunk
                }
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        return processed_chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents and generate embeddings"""
        processed_docs = []
        
        for i, doc in enumerate(tqdm(documents, desc="Processing documents")):
            try:
                # Generate embedding for document text
                embedding = self.generate_embedding(doc['text'])
                
                # Create metadata
                metadata = {
                    'source': doc.get('source', ''),
                    'type': doc.get('type', ''),
                    'chunk_type': 'document'
                }
                
                processed_doc = {
                    'id': f"doc_{i}",
                    'text': doc['text'],
                    'embedding': embedding,
                    'metadata': metadata,
                    'original_data': doc
                }
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue
        
        return processed_docs
    
    def _create_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Create text representation of a chunk for embedding"""
        text_parts = []
        
        # Add summary text if available
        if 'summary_text' in chunk:
            text_parts.append(chunk['summary_text'])
        
        # Add period information
        text_parts.append(f"Period: {chunk.get('period', 'Unknown')}")
        text_parts.append(f"Period type: {chunk.get('period_type', 'Unknown')}")
        
        # Add key metrics
        if 'total_sales' in chunk:
            text_parts.append(f"Total sales: ${chunk['total_sales']:,.2f}")
        if 'total_quantity' in chunk:
            text_parts.append(f"Total quantity: {chunk['total_quantity']}")
        if 'unique_products' in chunk:
            text_parts.append(f"Unique products: {chunk['unique_products']}")
        if 'unique_regions' in chunk:
            text_parts.append(f"Unique regions: {chunk['unique_regions']}")
        
        # Add sample data points if available
        if 'data' in chunk and len(chunk['data']) > 0:
            sample_data = chunk['data'][:5]  # First 5 records
            text_parts.append("Sample data points:")
            for record in sample_data:
                if 'product' in record and 'sales' in record:
                    text_parts.append(f"  - {record['product']}: ${record['sales']:,.2f}")
        
        return " ".join(text_parts)
    
    def store_in_chroma(self, processed_items: List[Dict[str, Any]]):
        """Store processed items in ChromaDB"""
        if not processed_items:
            logger.warning("No items to store in ChromaDB")
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for item in processed_items:
            ids.append(item['id'])
            texts.append(item['text'])
            embeddings.append(item['embedding'])
            metadatas.append(item['metadata'])
        
        # Add to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(processed_items)} items in ChromaDB")
        except Exception as e:
            logger.error(f"Error storing items in ChromaDB: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar items in ChromaDB"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def process_all_data(self, processed_data_file: str = "models/processed_data.json"):
        """Main method to process all data and store in ChromaDB"""
        # Load processed data
        data_file = Path(processed_data_file)
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")
        
        with open(data_file, 'r') as f:
            processed_data = json.load(f)
        
        logger.info("Starting embedding generation and storage...")
        
        # Process chunks
        chunks = processed_data.get('chunks', [])
        if chunks:
            logger.info(f"Processing {len(chunks)} chunks...")
            processed_chunks = self.process_chunks(chunks)
            self.store_in_chroma(processed_chunks)
        
        # Process documents
        documents = processed_data.get('documents', [])
        if documents:
            logger.info(f"Processing {len(documents)} documents...")
            processed_docs = self.process_documents(documents)
            self.store_in_chroma(processed_docs)
        
        # Save embedding info
        embedding_info = {
            'embedding_model': self.embedding_model,
            'total_chunks': len(chunks),
            'total_documents': len(documents),
            'chroma_persist_dir': str(self.chroma_persist_dir),
            'processed_at': str(np.datetime64('now'))
        }
        
        info_file = Path("models/embedding_info.json")
        with open(info_file, 'w') as f:
            json.dump(embedding_info, f, indent=2)
        
        logger.info(f"Embedding processing completed. Info saved to {info_file}")
        logger.info(f"Total items stored in ChromaDB: {len(chunks) + len(documents)}")

if __name__ == "__main__":
    # Run embedding generation
    embedding_manager = EmbeddingManager(embedding_model="sbert")
    embedding_manager.process_all_data()
    print("Embedding generation completed successfully!") 
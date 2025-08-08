"""
Simplified Embeddings Module for Sales Forecasting AI
This version works without sentence-transformers and chromadb for testing purposes.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingManager:
    """Simplified embedding manager for testing purposes."""
    
    def __init__(self, embedding_model: str = "simple", chroma_persist_dir: str = "models/chroma_db"):
        """
        Initialize the simplified embedding manager.
        
        Args:
            embedding_model: Model type (default: "simple")
            chroma_persist_dir: Directory for storing embeddings
        """
        self.embedding_model = embedding_model
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SimpleEmbeddingManager with model: {embedding_model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text.
        This is a placeholder that returns a fixed-size vector.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        # Simple hash-based embedding for testing
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to 384-dimensional vector (like sentence-transformers)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) >= 384:
                break
            embedding.append(float(int(hash_hex[i:i+2], 16)) / 255.0)
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process sales data chunks and add embeddings.
        
        Args:
            chunks: List of data chunks
            
        Returns:
            List of processed chunks with embeddings
        """
        processed_chunks = []
        
        for chunk in chunks:
            # Create text representation
            text = self._create_chunk_text(chunk)
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Add embedding to chunk
            processed_chunk = chunk.copy()
            processed_chunk['embedding'] = embedding
            processed_chunk['text'] = text
            
            processed_chunks.append(processed_chunk)
        
        logger.info(f"Processed {len(processed_chunks)} chunks")
        return processed_chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process document texts and add embeddings.
        
        Args:
            documents: List of document data
            
        Returns:
            List of processed documents with embeddings
        """
        processed_documents = []
        
        for doc in documents:
            text = doc.get('text', '')
            embedding = self.generate_embedding(text)
            
            processed_doc = doc.copy()
            processed_doc['embedding'] = embedding
            
            processed_documents.append(processed_doc)
        
        logger.info(f"Processed {len(processed_documents)} documents")
        return processed_documents
    
    def _create_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """
        Create a text representation of a chunk for embedding.
        
        Args:
            chunk: Data chunk
            
        Returns:
            Text representation
        """
        text_parts = []
        
        # Add period information
        if 'period' in chunk:
            text_parts.append(f"Period: {chunk['period']}")
        
        # Add summary
        if 'summary' in chunk:
            text_parts.append(f"Summary: {chunk['summary']}")
        
        # Add metadata
        if 'metadata' in chunk:
            metadata = chunk['metadata']
            if 'region' in metadata:
                text_parts.append(f"Region: {metadata['region']}")
            if 'product' in metadata:
                text_parts.append(f"Product: {metadata['product']}")
            if 'date_range' in metadata:
                text_parts.append(f"Date Range: {metadata['date_range']}")
        
        return " | ".join(text_parts)
    
    def store_in_chroma(self, processed_items: List[Dict[str, Any]]):
        """
        Store embeddings in a simple JSON format (simulating ChromaDB).
        
        Args:
            processed_items: List of processed items with embeddings
        """
        # Create a simple storage format
        storage_data = {
            'embeddings': [],
            'metadata': [],
            'documents': []
        }
        
        for item in processed_items:
            storage_data['embeddings'].append(item.get('embedding', []))
            storage_data['metadata'].append(item.get('metadata', {}))
            storage_data['documents'].append(item.get('text', ''))
        
        # Save to JSON file
        storage_file = self.chroma_persist_dir / "embeddings.json"
        with open(storage_file, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        logger.info(f"Stored {len(processed_items)} embeddings in {storage_file}")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar items using simple cosine similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar items
        """
        # Load stored embeddings
        storage_file = self.chroma_persist_dir / "embeddings.json"
        if not storage_file.exists():
            logger.warning("No embeddings found. Run process_all_data first.")
            return []
        
        with open(storage_file, 'r') as f:
            storage_data = json.load(f)
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, stored_embedding in enumerate(storage_data['embeddings']):
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            similarities.append({
                'index': i,
                'similarity': similarity,
                'metadata': storage_data['metadata'][i],
                'document': storage_data['documents'][i]
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:n_results]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def process_all_data(self, processed_data_file: str = "models/processed_data.json"):
        """
        Load processed data and generate embeddings.
        
        Args:
            processed_data_file: Path to processed data file
        """
        # Load processed data
        with open(processed_data_file, 'r') as f:
            processed_data = json.load(f)
        
        chunks = processed_data.get('chunks', [])
        documents = processed_data.get('documents', [])
        
        logger.info(f"Processing {len(chunks)} chunks and {len(documents)} documents")
        
        # Process chunks
        processed_chunks = self.process_chunks(chunks)
        
        # Process documents
        processed_documents = self.process_documents(documents)
        
        # Store in ChromaDB (simulated)
        all_items = processed_chunks + processed_documents
        self.store_in_chroma(all_items)
        
        logger.info("Embedding generation completed successfully!")

def main():
    """Main function to test the simplified embeddings module."""
    # Initialize embedding manager
    embedding_manager = SimpleEmbeddingManager()
    
    # Process all data
    embedding_manager.process_all_data()
    
    # Test search functionality
    test_query = "What were the sales trends in Q1 2024?"
    results = embedding_manager.search_similar(test_query, n_results=3)
    
    print(f"\nSearch results for: '{test_query}'")
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity']:.3f}")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Document: {result['document'][:100]}...")
        print()

if __name__ == "__main__":
    main()

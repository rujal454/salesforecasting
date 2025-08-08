"""
Phase 3: Retrieval-Augmented Answering
User enters natural language query, system retrieves top-k similar chunks,
passes chunks + query to LLM via prompt template
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import openai
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 embedding_manager=None,
                 llm_model: str = "gpt-3.5-turbo",
                 max_context_length: int = 4000):
        self.embedding_manager = embedding_manager
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        
        # Initialize OpenAI client
        self._init_openai()
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.llm_model}")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different types of queries"""
        templates = {
            "sales_analysis": """You are a sales forecasting expert. Based on the following historical sales data, answer the user's question.

Context Data:
{context}

User Question: {question}

Please provide a comprehensive answer based on the data provided. Include specific numbers, trends, and insights where available. If the data doesn't contain enough information to answer the question completely, acknowledge this and provide what insights you can.

Answer:""",
            
            "forecasting": """You are a sales forecasting expert. Based on the following historical sales data, provide a forecast or prediction.

Context Data:
{context}

User Question: {question}

Please provide a forecast based on the historical patterns. Include:
1. Your prediction
2. Confidence level
3. Key factors influencing your forecast
4. Any assumptions you're making

Answer:""",
            
            "trend_analysis": """You are a sales data analyst. Analyze the following sales data to identify trends and patterns.

Context Data:
{context}

User Question: {question}

Please provide a trend analysis including:
1. Key trends identified
2. Seasonal patterns (if any)
3. Growth or decline rates
4. Recommendations based on trends

Answer:""",
            
            "general": """You are a sales data expert. Answer the following question based on the provided sales data.

Context Data:
{context}

User Question: {question}

Provide a clear, data-driven answer. Use specific numbers and examples from the data when possible.

Answer:"""
        }
        return templates
    
    def classify_query(self, query: str) -> str:
        """Classify the type of query to select appropriate prompt template"""
        query_lower = query.lower()
        
        # Keywords for different query types
        forecasting_keywords = ['forecast', 'predict', 'next', 'future', 'upcoming', 'projection']
        trend_keywords = ['trend', 'pattern', 'growth', 'decline', 'increase', 'decrease', 'change']
        analysis_keywords = ['analyze', 'analysis', 'insight', 'performance', 'compare']
        
        if any(keyword in query_lower for keyword in forecasting_keywords):
            return "forecasting"
        elif any(keyword in query_lower for keyword in trend_keywords):
            return "trend_analysis"
        elif any(keyword in query_lower for keyword in analysis_keywords):
            return "sales_analysis"
        else:
            return "general"
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using vector similarity search"""
        if not self.embedding_manager:
            raise ValueError("Embedding manager not initialized")
        
        return self.embedding_manager.search_similar(query, n_results)
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            context_parts.append(f"=== Chunk {i+1} ===")
            context_parts.append(f"Period: {chunk['metadata'].get('period', 'Unknown')}")
            context_parts.append(f"Type: {chunk['metadata'].get('chunk_type', 'Unknown')}")
            context_parts.append(f"Content: {chunk['text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def truncate_context(self, context: str, query: str) -> str:
        """Truncate context to fit within token limits"""
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = self.max_context_length * 4
        
        # Reserve space for query and prompt template
        reserved_chars = 1000  # For query and prompt template
        available_chars = max_chars - reserved_chars
        
        if len(context) <= available_chars:
            return context
        
        # Truncate context
        truncated = context[:available_chars]
        
        # Try to truncate at a reasonable boundary
        last_period = truncated.rfind('.')
        if last_period > available_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[:last_period + 1]
        
        return truncated + "\n[Context truncated due to length limits]"
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using LLM"""
        try:
            # Classify query type
            query_type = self.classify_query(query)
            
            # Get appropriate prompt template
            template = self.prompt_templates.get(query_type, self.prompt_templates["general"])
            
            # Format prompt
            prompt = template.format(
                context=context,
                question=query
            )
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful sales forecasting assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'query_type': query_type,
                'model_used': self.llm_model,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': f"Sorry, I encountered an error while processing your request: {str(e)}",
                'query_type': 'error',
                'model_used': self.llm_model,
                'tokens_used': None
            }
    
    def answer_query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Main method to answer a user query using RAG"""
        try:
            # Step 1: Retrieve relevant chunks
            logger.info(f"Retrieving relevant chunks for query: {query}")
            chunks = self.retrieve_relevant_chunks(query, n_results)
            
            if not chunks:
                return {
                    'answer': "I couldn't find any relevant data to answer your question. Please try rephrasing your query or ask about different aspects of the sales data.",
                    'chunks_retrieved': 0,
                    'query_type': 'no_results'
                }
            
            # Step 2: Create context from chunks
            context = self.create_context_from_chunks(chunks)
            
            # Step 3: Truncate context if needed
            context = self.truncate_context(context, query)
            
            # Step 4: Generate response
            logger.info("Generating response using LLM")
            response = self.generate_response(query, context)
            
            # Step 5: Prepare final result
            result = {
                'query': query,
                'answer': response['answer'],
                'chunks_retrieved': len(chunks),
                'query_type': response['query_type'],
                'model_used': response['model_used'],
                'tokens_used': response['tokens_used'],
                'supporting_chunks': chunks[:3],  # Include top 3 chunks for transparency
                'context_length': len(context)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return {
                'query': query,
                'answer': f"An error occurred while processing your query: {str(e)}",
                'chunks_retrieved': 0,
                'query_type': 'error',
                'error': str(e)
            }
    
    def batch_answer_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple queries in batch"""
        results = []
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            result = self.answer_query(query)
            results.append(result)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get collection count
            collection_count = self.embedding_manager.collection.count()
            
            return {
                'total_documents': collection_count,
                'model_used': self.llm_model,
                'embedding_model': self.embedding_manager.embedding_model,
                'max_context_length': self.max_context_length
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}

# Example usage and testing
def test_rag_system():
    """Test the RAG system with sample queries"""
    from embeddings import EmbeddingManager
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(embedding_model="sbert")
    
    # Initialize RAG system
    rag_system = RAGSystem(embedding_manager=embedding_manager)
    
    # Test queries
    test_queries = [
        "What were the top performing products in Q3 2023?",
        "Show me sales trends for the Northeast region",
        "Predict sales for next quarter based on historical data",
        "Which products have declining sales trends?",
        "What is the average sales per region?"
    ]
    
    print("Testing RAG System...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag_system.answer_query(query)
        print(f"Answer: {result['answer']}")
        print(f"Chunks retrieved: {result['chunks_retrieved']}")
        print("-" * 30)

if __name__ == "__main__":
    # Test the RAG system
    test_rag_system() 
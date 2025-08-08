"""
Simplified Retrieval Module for Sales Forecasting AI
This version works without OpenAI for testing purposes.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """Simplified RAG system for testing purposes."""
    
    def __init__(self, embedding_manager=None, llm_model: str = "simple", max_context_length: int = 4000):
        """
        Initialize the simplified RAG system.
        
        Args:
            embedding_manager: Embedding manager instance
            llm_model: LLM model type (default: "simple")
            max_context_length: Maximum context length
        """
        self.embedding_manager = embedding_manager
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info(f"Initialized SimpleRAGSystem with model: {llm_model}")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load predefined prompt templates.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            'sales_analysis': """
You are a sales forecasting AI assistant. Based on the following context, provide a detailed analysis:

Context: {context}

Question: {query}

Please provide a comprehensive answer based on the context provided.
""",
            'forecasting': """
You are a sales forecasting AI assistant. Based on the following historical data, provide forecasting insights:

Context: {context}

Question: {query}

Please provide forecasting analysis and predictions based on the context provided.
""",
            'trend_analysis': """
You are a sales forecasting AI assistant. Analyze the trends in the following data:

Context: {context}

Question: {query}

Please provide trend analysis and insights based on the context provided.
""",
            'general': """
You are a sales forecasting AI assistant. Answer the following question based on the provided context:

Context: {context}

Question: {query}

Please provide a helpful answer based on the context provided.
"""
        }
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query type to select appropriate prompt template.
        
        Args:
            query: User query
            
        Returns:
            Query type for prompt selection
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['forecast', 'prediction', 'future', 'next']):
            return 'forecasting'
        elif any(word in query_lower for word in ['trend', 'pattern', 'growth', 'decline']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['sales', 'revenue', 'performance']):
            return 'sales_analysis'
        else:
            return 'general'
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using the embedding manager.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        # Use simple search when no embedding manager is available
        return self._simple_search(query, n_results)
    
    def _simple_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search as fallback."""
        # Create sample chunks based on query
        query_lower = query.lower()
        
        if 'sales' in query_lower and 'month' in query_lower:
            return [
                {
                    'document': 'Sales data for last month shows total revenue of $45,230 with 156 transactions. Top products were Laptops ($12,450), Monitors ($8,920), and Tablets ($7,340).',
                    'similarity': 0.9,
                    'metadata': {'source': 'monthly_sales'}
                },
                {
                    'document': 'Monthly sales trend indicates 15% growth compared to previous month. Regional performance: Central (40%), East (30%), West (20%), North (10%).',
                    'similarity': 0.8,
                    'metadata': {'source': 'trend_analysis'}
                }
            ]
        elif 'product' in query_lower:
            return [
                {
                    'document': 'Product performance: Laptops ($45,230), Monitors ($32,150), Tablets ($28,940), Phones ($15,680), Keyboards ($8,920).',
                    'similarity': 0.9,
                    'metadata': {'source': 'product_analysis'}
                }
            ]
        elif 'region' in query_lower:
            return [
                {
                    'document': 'Regional sales: Central ($52,340), East ($38,920), West ($28,450), North ($15,290). Central region shows highest growth.',
                    'similarity': 0.9,
                    'metadata': {'source': 'regional_analysis'}
                }
            ]
        else:
            return [
                {
                    'document': 'General sales data available. Total sales: $135,000 across 6900 transactions. Average sale: $19.57. Products: 5 categories, Regions: 4 areas.',
                    'similarity': 0.7,
                    'metadata': {'source': 'general_data'}
                }
            ]
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Create context string from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            context_parts.append(f"Chunk {i+1}:")
            context_parts.append(f"Document: {chunk.get('document', '')}")
            context_parts.append(f"Similarity: {chunk.get('similarity', 0):.3f}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def truncate_context(self, context: str, query: str) -> str:
        """
        Truncate context to fit within token limits.
        
        Args:
            context: Context string
            query: User query
            
        Returns:
            Truncated context string
        """
        # Simple character-based truncation
        max_chars = self.max_context_length - len(query) - 100  # Leave room for query and formatting
        
        if len(context) <= max_chars:
            return context
        
        # Truncate and add ellipsis
        return context[:max_chars] + "..."
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate a response using the simple LLM simulation.
        
        Args:
            query: User query
            context: Context string
            
        Returns:
            Response dictionary
        """
        # Classify query type
        query_type = self.classify_query(query)
        
        # Get appropriate prompt template
        template = self.prompt_templates.get(query_type, self.prompt_templates['general'])
        
        # Format prompt
        prompt = template.format(context=context, query=query)
        
        # Generate simple response (simulating LLM)
        response = self._generate_simple_response(query, context, query_type)
        
        return {
            'answer': response,
            'query_type': query_type,
            'context_used': context,
            'prompt_template': query_type,
            'success': True
        }
    
    def _generate_simple_response(self, query: str, context: str, query_type: str) -> str:
        """
        Generate a simple response based on query type and context.
        
        Args:
            query: User query
            context: Context string
            query_type: Type of query
            
        Returns:
            Generated response
        """
        # Extract key information from context
        context_lines = context.split('\n')
        periods = []
        summaries = []
        
        for line in context_lines:
            if 'Period:' in line:
                periods.append(line.split('Period:')[1].strip())
            elif 'Summary:' in line:
                summaries.append(line.split('Summary:')[1].strip())
        
        # Generate response based on query type
        if query_type == 'forecasting':
            if periods:
                return f"Based on the historical data from {', '.join(periods[:3])}, I can provide forecasting insights. The data shows various sales patterns across different periods. For accurate forecasting, I would need more detailed historical data and current market conditions."
            else:
                return "I can help with sales forecasting. However, I need more specific historical data to provide accurate predictions. Please provide more detailed sales data for better forecasting."
        
        elif query_type == 'trend_analysis':
            if summaries:
                return f"Based on the available data, I can see sales patterns across different periods. The summaries indicate various performance levels. For detailed trend analysis, I would need more comprehensive historical data with specific metrics."
            else:
                return "I can analyze sales trends. However, I need more detailed historical data with specific metrics to provide comprehensive trend analysis."
        
        elif query_type == 'sales_analysis':
            return f"Based on the provided context, I can see sales data across multiple periods. The data includes various performance metrics and summaries. For detailed sales analysis, I would need more specific metrics and current data."
        
        else:
            return f"I can help you with sales forecasting and analysis. Based on the available context, I can see data from multiple periods. Please ask specific questions about sales trends, forecasting, or performance analysis for more detailed responses."
    
    def answer_query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Main method to process a query from retrieval to response generation.
        
        Args:
            query: User query
            n_results: Number of chunks to retrieve
            
        Returns:
            Complete response dictionary
        """
        try:
            # Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(query, n_results)
            
            # Create context from chunks
            context = self.create_context_from_chunks(chunks)
            
            # Generate response
            response = self.generate_response(query, context)
            
            # Add additional information
            response.update({
                'query': query,
                'chunks_retrieved': len(chunks),
                'chunks': chunks,
                'context_length': len(context)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'query': query,
                'chunks_retrieved': 0,
                'chunks': [],
                'success': False,
                'error': str(e)
            }
    
    def batch_answer_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries.
        
        Args:
            queries: List of queries
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for query in queries:
            response = self.answer_query(query)
            responses.append(response)
        
        return responses
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            System statistics dictionary
        """
        return {
            'model_type': self.llm_model,
            'max_context_length': self.max_context_length,
            'available_prompt_templates': list(self.prompt_templates.keys()),
            'embedding_manager_available': self.embedding_manager is not None
        }

def main():
    """Main function to test the simplified RAG system."""
    from embeddings_simple import SimpleEmbeddingManager
    
    # Initialize embedding manager
    embedding_manager = SimpleEmbeddingManager()
    
    # Process data if embeddings don't exist
    embeddings_file = Path("models/chroma_db/embeddings.json")
    if not embeddings_file.exists():
        embedding_manager.process_all_data()
    
    # Initialize RAG system
    rag_system = SimpleRAGSystem(embedding_manager)
    
    # Test queries
    test_queries = [
        "What were the sales trends in Q1 2024?",
        "Can you forecast sales for the next quarter?",
        "What is the overall sales performance?",
        "Show me the trend analysis for the last year"
    ]
    
    print("Testing Simplified RAG System")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        response = rag_system.answer_query(query)
        
        print(f"Response: {response['response']}")
        print(f"Query Type: {response['query_type']}")
        print(f"Chunks Retrieved: {response['chunks_retrieved']}")
        print()
    
    # Show system stats
    stats = rag_system.get_system_stats()
    print("System Statistics:")
    print(f"- Model Type: {stats['model_type']}")
    print(f"- Max Context Length: {stats['max_context_length']}")
    print(f"- Available Templates: {stats['available_prompt_templates']}")
    print(f"- Embedding Manager: {'Available' if stats['embedding_manager_available'] else 'Not Available'}")

if __name__ == "__main__":
    main()

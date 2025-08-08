"""
Real AI Retrieval Module for Sales Forecasting AI
This version uses OpenAI GPT for intelligent responses.
"""

import json
import logging
import os
from typing import List, Dict, Any
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealAISystem:
    """Real AI system using OpenAI GPT."""
    
    def __init__(self, embedding_manager=None, llm_model: str = "gpt-3.5-turbo", max_context_length: int = 4000):
        """
        Initialize the real AI system.
        
        Args:
            embedding_manager: Embedding manager instance
            llm_model: LLM model type (default: "gpt-3.5-turbo")
            max_context_length: Maximum context length
        """
        self.embedding_manager = embedding_manager
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        
        # Initialize OpenAI
        self._init_openai()
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info(f"Initialized RealAISystem with model: {llm_model}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        
        openai.api_key = api_key
        logger.info("OpenAI client initialized")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load predefined prompt templates for different query types.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            'sales_analysis': """You are an expert sales forecasting AI assistant. Analyze the following sales data and provide detailed insights.

Context Data: {context}

Question: {query}

Please provide a comprehensive analysis including:
- Key insights from the data
- Specific numbers and trends
- Actionable recommendations
- Data-driven conclusions

Answer:""",
            
            'forecasting': """You are an expert sales forecasting AI assistant. Based on the historical data provided, give forecasting insights.

Context Data: {context}

Question: {query}

Please provide:
- Trend analysis
- Forecasting predictions
- Confidence levels
- Key factors affecting sales
- Recommendations for improvement

Answer:""",
            
            'trend_analysis': """You are an expert sales forecasting AI assistant. Analyze the trends in the provided data.

Context Data: {context}

Question: {query}

Please provide:
- Trend identification
- Pattern analysis
- Growth/decline insights
- Seasonal patterns
- Comparative analysis

Answer:""",
            
            'general': """You are an expert sales forecasting AI assistant. Answer the following question based on the provided sales data.

Context Data: {context}

Question: {query}

Please provide a helpful, data-driven answer based on the context provided.

Answer:"""
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
        
        if any(word in query_lower for word in ['forecast', 'prediction', 'future', 'next', 'upcoming']):
            return 'forecasting'
        elif any(word in query_lower for word in ['trend', 'pattern', 'growth', 'decline', 'change']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['sales', 'revenue', 'performance', 'analysis']):
            return 'sales_analysis'
        else:
            return 'general'
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using embedding manager.
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant chunks
        """
        if self.embedding_manager:
            return self.embedding_manager.search_similar(query, n_results)
        else:
            # Fallback to simple keyword matching
            return self._simple_search(query, n_results)
    
    def _simple_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search as fallback."""
        # This is a simplified search - in real implementation, use embeddings
        return [
            {
                'document': f"Sample sales data chunk for query: {query}",
                'similarity': 0.8,
                'metadata': {'source': 'sample_data'}
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
        if not chunks:
            return "No relevant data found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Data Chunk {i}:\n{chunk['document']}")
        
        return "\n\n".join(context_parts)
    
    def truncate_context(self, context: str, query: str) -> str:
        """
        Truncate context to fit within token limits.
        
        Args:
            context: Full context string
            query: User query
            
        Returns:
            Truncated context string
        """
        # Simple truncation - in production, use proper token counting
        max_context_length = self.max_context_length - len(query) - 100
        
        if len(context) <= max_context_length:
            return context
        
        # Truncate and add ellipsis
        return context[:max_context_length] + "..."
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate response using OpenAI GPT.
        
        Args:
            query: User query
            context: Context string
            
        Returns:
            Response dictionary
        """
        try:
            # Classify query type
            query_type = self.classify_query(query)
            
            # Get appropriate prompt template
            prompt_template = self.prompt_templates[query_type]
            
            # Format prompt
            formatted_prompt = prompt_template.format(
                context=context,
                query=query
            )
            
            # Truncate context if needed
            formatted_prompt = self.truncate_context(formatted_prompt, query)
            
            # Call OpenAI
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert sales forecasting AI assistant. Provide accurate, data-driven insights."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'query_type': query_type,
                'model_used': self.llm_model,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}. Please check your OpenAI API key and try again.",
                'query_type': 'error',
                'model_used': self.llm_model,
                'tokens_used': None,
                'success': False,
                'error': str(e)
            }
    
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
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            
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
        Get system statistics and status.
        
        Returns:
            System statistics dictionary
        """
        return {
            'model': self.llm_model,
            'max_context_length': self.max_context_length,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'embedding_manager': bool(self.embedding_manager),
            'prompt_templates': list(self.prompt_templates.keys())
        }

def main():
    """Test the real AI system."""
    try:
        # Initialize system
        rag_system = RealAISystem()
        
        # Test queries
        test_queries = [
            "What are the top performing products?",
            "Can you forecast sales for next quarter?",
            "Show me sales trends by region"
        ]
        
        print("🤖 Real AI System Test")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = rag_system.answer_query(query)
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Model: {response['model_used']}")
            print(f"Success: {response['success']}")
        
        # System stats
        stats = rag_system.get_system_stats()
        print(f"\nSystem Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have:")
        print("1. OpenAI API key in .env file")
        print("2. Installed openai package: pip install openai")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main pipeline script to run all phases of the sales forecasting system
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env") 
print("LOADED KEY:", os.getenv("OPENAI_API_KEY"))
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_ingestion import DataIngestion
from embeddings import EmbeddingManager
from retrieval import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase_1():
    """Run Phase 1: Data Ingestion & Processing"""
    logger.info("=" * 50)
    logger.info("PHASE 1: Data Ingestion & Processing")
    logger.info("=" * 50)
    
    try:
        ingestion = DataIngestion()
        processed_data = ingestion.process_all_data()
        logger.info("✅ Phase 1 completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        return False

def run_phase_2():
    """Run Phase 2: Embedding & Vector Storage"""
    logger.info("=" * 50)
    logger.info("PHASE 2: Embedding & Vector Storage")
    logger.info("=" * 50)
    
    try:
        embedding_manager = EmbeddingManager(embedding_model="sbert")
        embedding_manager.process_all_data()
        logger.info("✅ Phase 2 completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        return False

def run_phase_3():
    """Run Phase 3: Test RAG System"""
    logger.info("=" * 50)
    logger.info("PHASE 3: Testing RAG System")
    logger.info("=" * 50)
    
    try:
        embedding_manager = EmbeddingManager(embedding_model="sbert")
        rag_system = RAGSystem(embedding_manager=embedding_manager)
        
        # Test queries
        test_queries = [
            "What were the top performing products?",
            "Show me sales trends",
            "Which region has the highest sales?"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            response = rag_system.answer_query(query)
            logger.info(f"Response: {response['answer'][:100]}...")
        
        logger.info("✅ Phase 3 completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        return False

def run_phase_4():
    """Run Phase 4: Start UI"""
    logger.info("=" * 50)
    logger.info("PHASE 4: Starting UI")
    logger.info("=" * 50)
    
    try:
        logger.info("Starting Streamlit app...")
        logger.info("Run: streamlit run ui/streamlit_app.py")
        logger.info("Or run: python ui/gradio_app.py")
        logger.info("✅ Phase 4 instructions provided!")
        return True
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {e}")
        return False

def main():
    """Run the complete pipeline"""
    logger.info("🚀 Starting Sales Forecasting Pipeline")
    logger.info(f"Started at: {datetime.now()}")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_API_KEY not set. Some features may not work.")
    
    # Run phases
    phases = [
        ("Phase 1: Data Ingestion", run_phase_1),
        ("Phase 2: Embedding Generation", run_phase_2),
        ("Phase 3: RAG Testing", run_phase_3),
        ("Phase 4: UI Setup", run_phase_4)
    ]
    
    results = {}
    
    for phase_name, phase_func in phases:
        logger.info(f"\n🔄 Running {phase_name}...")
        success = phase_func()
        results[phase_name] = success
        
        if not success:
            logger.error(f"❌ {phase_name} failed. Stopping pipeline.")
            break
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    
    for phase_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{phase_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All phases completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start Streamlit UI: streamlit run ui/streamlit_app.py")
        logger.info("2. Or start Gradio UI: python ui/gradio_app.py")
        logger.info("3. Open your browser and start asking questions!")
    else:
        logger.error("\n❌ Some phases failed. Check the logs above.")
    
    logger.info(f"\nPipeline completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 
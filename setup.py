#!/usr/bin/env python3
"""
Setup script for Sales Forecasting AI
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models",
        "models/chroma_db",
        "notebooks",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./models/chroma_db

# Model Configuration
EMBEDDING_MODEL=sbert  # or "openai"
LLM_MODEL=gpt-3.5-turbo

# Application Configuration
MAX_CONTEXT_LENGTH=4000
DEFAULT_N_RESULTS=5
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please update OPENAI_API_KEY in .env file")
    else:
        print("✅ .env file already exists")

def run_initial_setup():
    """Run initial data processing"""
    print("🔄 Running initial setup...")
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        # Import and run data ingestion
        from data_ingestion import DataIngestion
        ingestion = DataIngestion()
        ingestion.process_all_data()
        
        print("✅ Initial data processing completed!")
        return True
    except Exception as e:
        print(f"❌ Error in initial setup: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Sales Forecasting AI")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return
    
    # Create .env file
    print("\n⚙️  Setting up configuration...")
    create_env_file()
    
    # Run initial setup
    print("\n🔄 Running initial data processing...")
    if not run_initial_setup():
        print("❌ Setup failed during initial data processing")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Update OPENAI_API_KEY in .env file (optional)")
    print("2. Start Streamlit UI: streamlit run ui/streamlit_app.py")
    print("3. Or start Gradio UI: python ui/gradio_app.py")
    print("4. Or run the complete pipeline: python run_pipeline.py")
    print("\nHappy forecasting! 📊")

if __name__ == "__main__":
    main() 
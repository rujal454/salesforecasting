#!/usr/bin/env python3
"""
Simple script to run both Flask backend and Streamlit frontend
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting Sales Forecasting AI...")
    print("=" * 50)
    
    # Create uploads directory
    Path("uploads").mkdir(exist_ok=True)
    
    print("📊 Starting Flask Backend...")
    flask_process = subprocess.Popen([
        sys.executable, "app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for Flask to start
    time.sleep(3)
    
    print("🎨 Starting Streamlit Frontend...")
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "streamlit_frontend.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for Streamlit to start
    time.sleep(5)
    
    print("\n" + "=" * 50)
    print("🎉 Application is running!")
    print("=" * 50)
    print("\n📊 Access URLs:")
    print("  🔗 Frontend: http://localhost:8501")
    print("  🔗 Backend:  http://localhost:5000")
    print("\n📋 Usage:")
    print("  1. Open http://localhost:8501 in your browser")
    print("  2. Upload a CSV file with sales data")
    print("  3. Generate forecasts and ask AI questions")
    print("\n🛑 Press Ctrl+C to stop both services")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if flask_process.poll() is not None:
                print("❌ Flask backend stopped")
                break
            if streamlit_process.poll() is not None:
                print("❌ Streamlit frontend stopped")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
        # Stop Flask
        if flask_process.poll() is None:
            flask_process.terminate()
            flask_process.wait(timeout=5)
        
        # Stop Streamlit
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
        
        print("✅ Services stopped")

if __name__ == "__main__":
    main()

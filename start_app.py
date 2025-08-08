#!/usr/bin/env python3
"""
Startup script for Sales Forecasting AI
Launches both Flask backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'streamlit', 'pandas', 'plotly', 'requests',
        'scikit-learn', 'transformers', 'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_flask_backend():
    """Start Flask backend"""
    print("🚀 Starting Flask backend...")
    
    try:
        # Start Flask app
        flask_process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for Flask to start
        time.sleep(3)
        
        # Check if Flask is running
        if flask_process.poll() is None:
            print("✅ Flask backend started successfully")
            print("🔗 Backend URL: http://localhost:5000")
            return flask_process
        else:
            stdout, stderr = flask_process.communicate()
            print("❌ Flask backend failed to start")
            print(f"Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Flask backend: {e}")
        return None

def start_streamlit_frontend():
    """Start Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    
    try:
        # Start Streamlit app
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_frontend.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for Streamlit to start
        time.sleep(5)
        
        # Check if Streamlit is running
        if streamlit_process.poll() is None:
            print("✅ Streamlit frontend started successfully")
            print("🔗 Frontend URL: http://localhost:8501")
            return streamlit_process
        else:
            stdout, stderr = streamlit_process.communicate()
            print("❌ Streamlit frontend failed to start")
            print(f"Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Streamlit frontend: {e}")
        return None

def cleanup_processes(processes):
    """Clean up running processes"""
    print("\n🛑 Shutting down processes...")
    
    for process in processes:
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Warning: Could not terminate process: {e}")

def main():
    """Main startup function"""
    print("🚀 Sales Forecasting AI - Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    processes = []
    
    try:
        # Start Flask backend
        flask_process = start_flask_backend()
        if not flask_process:
            print("❌ Failed to start Flask backend. Exiting.")
            return
        processes.append(flask_process)
        
        # Start Streamlit frontend
        streamlit_process = start_streamlit_frontend()
        if not streamlit_process:
            print("❌ Failed to start Streamlit frontend. Exiting.")
            cleanup_processes(processes)
            return
        processes.append(streamlit_process)
        
        print("\n" + "=" * 50)
        print("🎉 Sales Forecasting AI is running!")
        print("=" * 50)
        print("\n📊 Application URLs:")
        print("  🔗 Frontend: http://localhost:8501")
        print("  🔗 Backend:  http://localhost:5000")
        print("\n📋 Usage:")
        print("  1. Open http://localhost:8501 in your browser")
        print("  2. Upload a CSV file with sales data")
        print("  3. Generate forecasts and ask AI questions")
        print("\n🛑 Press Ctrl+C to stop the application")
        
        # Keep the application running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                for process in processes:
                    if process.poll() is not None:
                        print(f"❌ Process terminated unexpectedly")
                        return
                        
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        cleanup_processes(processes)
        print("✅ Application stopped")

if __name__ == "__main__":
    main()

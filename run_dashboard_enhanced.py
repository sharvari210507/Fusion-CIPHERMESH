#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher for CIPHERMESH
Launches the modern, professional Streamlit dashboard with Amazon-like UI
"""

import subprocess
import sys
import os

def main():
    """Launch the enhanced Streamlit dashboard."""
    print("🚀 Launching CIPHERMESH Enhanced Dashboard...")
    print("🔐 Privacy-Preserving Fraud Signal Sharing Across Banks")
    print("📊 Modern UI with Amazon-inspired design")
    print("-" * 50)

    # Change to the project directory
    project_dir = "/home/sharvari/Fusion-CIPHERMESH"
    os.chdir(project_dir)

    # Activate virtual environment and run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard_enhanced.py",
        "--server.port=8501",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    print("🌐 Dashboard will be available at:")
    print("   Local:    http://localhost:8501")
    print("   Network:  http://10.56.161.155:8501")
    print("   External: http://152.58.33.115:8501")
    print("-" * 50)
    print("💡 Press Ctrl+C to stop the dashboard")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
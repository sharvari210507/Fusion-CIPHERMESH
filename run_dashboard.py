#!/usr/bin/env python3
"""
Launcher script for the CIPHERMESH dashboard.
"""

import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run the streamlit dashboard
    print("Starting CIPHERMESH Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()
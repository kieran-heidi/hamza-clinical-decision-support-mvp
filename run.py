#!/usr/bin/env python3
"""
Railway deployment wrapper for Streamlit.
Reads PORT from environment and runs Streamlit on that port.
"""
import os
import sys
import subprocess

# Get PORT from environment (Railway sets this)
port = os.environ.get("PORT", "8080")

# Ensure port is an integer
try:
    port = int(port)
except ValueError:
    print(f"Warning: PORT environment variable '{port}' is not a valid integer. Using 8080.")
    port = 8080

# Set Streamlit config via environment variables
os.environ["STREAMLIT_SERVER_PORT"] = str(port)
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

print(f"Starting Streamlit on port {port}...")

# Run Streamlit
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
]

sys.exit(subprocess.run(cmd).returncode)


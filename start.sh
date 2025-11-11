#!/bin/bash
# Start script for Railway deployment
# This ensures the PORT environment variable is properly expanded

PORT=${PORT:-8080}
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0


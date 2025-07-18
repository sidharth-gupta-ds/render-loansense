#!/bin/bash

# Run Streamlit frontend

echo "🎨 Starting Streamlit Frontend..."

# Start Streamlit app using the configured Python environment
streamlit run streamlit_ui.py --server.port 8501

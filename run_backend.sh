#!/bin/bash

# Run FastAPI backend

echo "🚀 Starting FastAPI Backend..."

# Change to backend directory
cd backend

# Start FastAPI server using the configured Python environment
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

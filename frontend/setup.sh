#!/bin/bash

# Setup script for Loan Prediction System

echo "🚀 Setting up Loan Prediction System..."

PORT=8501

echo "🔍 Checking if port $PORT is in use..."

PID=$(lsof -ti :$PORT)

if [ -n "$PID" ]; then
  echo "⚠️ Port $PORT is in use by process $PID. Killing it..."
  kill -9 $PID
  echo "✅ Process $PID killed. Port $PORT is now free."
else
  echo "✅ Port $PORT is free."
fi

# Start both backend and frontend in parallel
echo "🚀 Starting Backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

sleep 2
echo "🚀 Starting Frontend..."
streamlit run streamlit_ui.py --server.port 8501

echo "✅ Both Backend and Frontend started!"
echo ""
echo "📚 API Docs at: http://localhost:8000/docs"
echo "🎨 Streamlit UI at: http://localhost:8501"



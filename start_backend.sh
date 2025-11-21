#!/bin/bash
# Start the FastAPI backend server

echo "Starting SAM3 Backend Server on port 8000..."
echo "Make sure you have activated your conda environment first:"
echo "  conda activate sam3"
echo ""

cd backend
python app.py

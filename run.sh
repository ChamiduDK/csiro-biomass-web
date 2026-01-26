#!/bin/bash

echo "========================================"
echo "CSIRO Biomass Prediction Web App"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "And: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment!"
    exit 1
fi

echo ""
echo "Starting Flask application..."
echo "Web interface will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python app.py

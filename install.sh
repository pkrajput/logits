#!/bin/bash

# Installation script for Code Contests dataset experiment

echo "=========================================="
echo "Setting up Code Contests Experiment"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated."
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a .env file with your OpenAI API key:"
echo "   OPENAI_API_KEY=your-api-key-here"
echo ""
echo "2. Run the experiment:"
echo "   python3 calculateLogits.py"
echo ""
echo "The Code Contests dataset will be automatically"
echo "downloaded on first run (may take a few minutes)."
echo ""

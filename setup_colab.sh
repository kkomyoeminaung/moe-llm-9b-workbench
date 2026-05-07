#!/bin/bash
# setup_colab.sh - Auto-setup script for Google Colab

echo "🚀 Starting MoE LLM Workbench Setup..."

# 1. Install System Dependencies
sudo apt-get update
sudo apt-get install -y sqlite3

# 2. Install Python Dependencies
echo "🐍 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 3. Setup Node.js (for Frontend)
echo "📦 Installing Node dependencies..."
npm install

# 4. Create necessary directories
mkdir -p exports
mkdir -p checkpoints
mkdir -p data/rag_index

echo "✅ Setup Complete!"

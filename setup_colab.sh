#!/bin/bash
# setup_colab.sh - Auto-setup script for Google Colab / Kaggle

echo "🚀 Starting MoE LLM Workbench Setup..."

# 1. Install System Dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update &> /dev/null
sudo apt-get install -y sqlite3 nodejs npm &> /dev/null

# 2. Install Python Dependencies
echo "🐍 Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt &> /dev/null

# 3. Setup Node.js & Node Modules
echo "📦 Installing Node dependencies..."
npm install &> /dev/null
npm install -g localtunnel &> /dev/null

# 4. Create necessary directories
echo "📁 Preparing directory structure..."
mkdir -p exports
mkdir -p checkpoints
mkdir -p data/rag_index
mkdir -p models

echo "✅ Setup Complete!"

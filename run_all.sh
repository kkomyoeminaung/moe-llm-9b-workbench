#!/bin/bash
# Universal Launcher for 7B MoE System
# Supports: Local Laptop, Google Colab, Kaggle, Lightning AI, Hugging Face Spaces

# Automate environment setup
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from .env.example..."
    cp .env.example .env
fi

echo "========================================================="
echo "🚀 Booting Universal 7B MoE Architecture..."
echo "========================================================="

# Detect Environment
ENV_TYPE="Local"
if [ -n "$COLAB_GPU" ] || [ -n "$GOOGLE_COLLAB" ]; then
    ENV_TYPE="Google Colab"
elif [ -d "/kaggle/working" ]; then
    ENV_TYPE="Kaggle"
elif [ -d "/lightning" ] || [ -n "$LIGHTNING_APP_STATE_URL" ]; then
    ENV_TYPE="Lightning AI"
elif [ -n "$SPACE_ID" ]; then
    ENV_TYPE="Hugging Face Spaces"
fi

echo "🌍 Detected Runtime: $ENV_TYPE"

# 1. Install pip dependencies
echo "📦 Installing/Verifying Python Dependencies..."
pip install -r requirements.txt -q

# 1.5. Download Model (Deprecated: Backend handles auto-download)
# 2. Install Node dependencies if npm is available
if command -v npm &> /dev/null; then
    echo "📦 Installing Frontend Dependencies..."
    npm install
else
    echo "⚠️ npm not found. Skipping frontend build (Useful for backend-only API/Spaces)."
fi

# 3. Start Backend
echo "🔥 Starting 7B MoE Core Engine (Backend)..."
python3 backend/app_unified.py &
BACKEND_PID=$!

# Wait briefly for backend to start
sleep 3

# 4. Start Frontend
if command -v npm &> /dev/null; then
    echo "🌐 Starting UI (Frontend)..."
    export MOCK_AI=false
    npm run dev -- --host 0.0.0.0 &
    FRONTEND_PID=$!
fi

echo "✅ 7B MoE System Fully Initialized and Running seamlessly!"
echo "📡 Engine is ready to receive RAG data, self-learn, and process logic."

# Keep container/process alive
wait -n
exit $?

#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve &
sleep 5

# Check if logged in to Ollama Cloud
if [ -z "$OLLAMA_CLOUD_TOKEN" ] && [ ! -f /root/.ollama/config.json ]; then
  echo "You are not logged in to Ollama Cloud."
  echo ""
  echo "Please log in at https://ollama.ai/account"
  echo "   and copy your Cloud API Token."
  echo ""
  echo "Then rerun the container with:"
  echo ""
  echo "   docker run -p 8501:8501 -v ~/.ollama:/root/.ollama \\"
  echo "       -e OLLAMA_CLOUD_TOKEN=sk-xxxxxx agentic-net"
  echo ""
  exit 1
fi

# Pull the required model
echo "Pulling gpt-oss:20b-cloud model..."
ollama pull gpt-oss:20b-cloud || {
  echo "Failed to pull model. Make sure your Cloud login is valid."
  exit 1
}

echo "Starting Streamlit app..."
echo "Access it at: http://localhost:8501"

# Start Streamlit (ensure it binds to all interfaces)
exec streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.serverAddress=localhost
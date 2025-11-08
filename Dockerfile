FROM python:3.10-slim

# Update system + install curl
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Install Python dependencies
RUN pip install --no-cache-dir torch pandas numpy

# Set working directory
WORKDIR /job

# Default command
CMD ["python", "agent_train.py"]

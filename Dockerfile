FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Python dependencies
#RUN pip install --no-cache-dir torch pandas numpy streamlit ollama 
RUN pip install ----no-cache-dir -r /requirements.txt

WORKDIR /app
COPY . /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]

# Dockerfile
FROM python:3.11-slim

# Prevents prompts
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (if PyMuPDF needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy code first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Streamlit config for headless
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# For sentence-transformers on CPU
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Railway sets PORT dynamically
# Expose a port (Railway will map to the PORT env var)
EXPOSE 8080

# Use a shell script to properly read PORT environment variable
CMD sh -c 'streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0'

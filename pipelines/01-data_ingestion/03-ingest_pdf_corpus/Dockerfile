# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data \
             /app/scripts \
             /app/output/openalex \
             /app/output/core \
             /app/output/other \
             /app/output/valid_pdfs \
             /app/output/invalid_pdfs \
             /app/output/duplicate_pdfs

# Copy all Python scripts
COPY scripts/ scripts/

# Set permissions
RUN chmod +x scripts/*.py

# Default command (can be overridden in docker-compose)
CMD ["python", "scripts/01-openalex_fetcher.py"]
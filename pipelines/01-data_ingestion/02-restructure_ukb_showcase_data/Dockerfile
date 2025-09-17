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
RUN mkdir -p /app/scripts 

# Copy all Python scripts
COPY scripts/process_showcase.py ./scripts/

# Set permissions
RUN chmod +x scripts/process_showcase.py

# Default command (can be overridden in docker-compose)
CMD ["python", "scripts/process_showcase.py"]
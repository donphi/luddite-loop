# ============================================================================
# FILE: Dockerfile
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/Dockerfile
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Container for PDF corpus ingestion pipeline with all dependencies
# ============================================================================

# Base Image - Python 3.11 slim for smaller size and good compatibility
FROM python:3.11-slim

# Set working directory for application files
WORKDIR /app

# Install system dependencies needed for PDF processing and network requests
# Using --no-install-recommends to keep image size small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
# This allows Docker to reuse the dependency installation layer if requirements.txt hasn't changed
COPY requirements.txt .

# Install Python dependencies without cache to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories for input data, scripts, and output organization
RUN mkdir -p /app/data \
             /app/scripts \
             /app/output/openalex \
             /app/output/core \
             /app/output/other \
             /app/output/valid_pdfs \
             /app/output/invalid_pdfs \
             /app/output/duplicate_pdfs

# Copy all Python scripts to the container
COPY scripts/ scripts/

# Set permissions to make scripts executable
RUN chmod +x scripts/*.py

# Default command (can be overridden in docker-compose)
# Starts with OpenAlex fetcher as the first stage of the pipeline
CMD ["python", "scripts/01-openalex_fetcher.py"]
# ============================================================================
# FILE: Dockerfile
# LOCATION: pipelines/01-data_ingestion/02-restructure_ukb_showcase_data/Dockerfile
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Container for UK Biobank data restructuring with GPU-accelerated SPECTER 2 embeddings
# ============================================================================

# Base Image - CUDA 12.8 enabled for GPU acceleration
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set working directory for application files
WORKDIR /app

# Install Python 3 and system dependencies for data processing
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables for consistent Python execution and CUDA configuration
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create necessary directories for scripts, embeddings cache, and output
RUN mkdir -p /app/scripts /app/embedding_cache /output/embedding_cache

# Copy requirements first for Docker layer caching optimization
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support (latest stable for CUDA 12.x compatibility)
RUN pip3 install --no-cache-dir \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# Install Python requirements from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download SPECTER 2 model from HuggingFace to avoid runtime downloads
# This model is used for semantic similarity in acronym selection
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext'); \
    AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')"

# Optional: Install AllenNLP if needed for advanced NLP tasks (commented out)
# RUN pip3 install --no-cache-dir allennlp allennlp-models

# Copy all Python scripts to container
COPY scripts/*.py ./scripts/

# Set executable permissions on Python scripts
RUN chmod +x scripts/*.py

# Create symlink for python command (convenience)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Default command - runs the Meta-Inventory processing script with embeddings
CMD ["python3", "scripts/process_showcase_meta.py"]
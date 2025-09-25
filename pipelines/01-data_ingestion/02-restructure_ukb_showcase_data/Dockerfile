# Base Image - CUDA 12.8 enabled
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set working directory
WORKDIR /app

# Install Python 3 and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create necessary directories
RUN mkdir -p /app/scripts /app/embedding_cache /output/embedding_cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support (latest stable for CUDA 12.x)
RUN pip3 install --no-cache-dir \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# Install requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install SPECTER 2 and transformers
# Pre-download SPECTER 2 model from HuggingFace
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext'); \
    AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')"

# Optional: Install AllenNLP if needed (comment out if not using)
# RUN pip3 install --no-cache-dir allennlp allennlp-models

# Copy all Python scripts
COPY scripts/*.py ./scripts/

# Set permissions
RUN chmod +x scripts/*.py

# Create symlink for python command
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Default command
CMD ["python3", "scripts/process_showcase_meta.py"]
# Pipeline Documentation and Standardization Prompt

## Task Overview
You are tasked with analyzing and documenting a research pipeline folder. You must read every file, standardize comments, document functionality, and create a comprehensive README while maintaining the working condition of all code.

## Critical Rules
1. **DO NOT** change the working functionality of any file
2. **PRESERVE** all logic, imports, and execution flow
3. **ONLY** modify comments and documentation
4. **KEEP** original file names unless they don't follow standard naming conventions
5. **ADD** documentation without breaking existing code

## File Processing Instructions

### Step 1: File Analysis and Comment Standardization

For **EVERY** file in the folder:

#### Python Files (.py)
```python
# ============================================================================
# FILE: data_processor.py
# LOCATION: 01-data_ingestion/02-pdf_extraction/data_processor.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Processes PDF files and extracts text content for downstream analysis
# ============================================================================

"""
MODULE OVERVIEW:
This module handles PDF text extraction using PyPDF2 and processes
the extracted content for structured data conversion.

CLASSES:
- PDFProcessor: Main class for handling PDF operations
  
METHODS:
- extract_text(): Extracts raw text from PDF pages
- clean_content(): Removes special characters and normalizes text
- save_output(): Saves processed data to JSON format

ROUTES:
- N/A (This is a processing module, not a web service)

HYPERPARAMETERS:
- MAX_PAGES: 1000 (in extract_text method, line 45)
- CHUNK_SIZE: 512 (in clean_content method, line 78)
- MIN_TEXT_LENGTH: 100 (in validate_output method, line 92)

SEEDS:
- RANDOM_SEED: 42 (in initialize method, line 23)
- Used for: Consistent text sampling when file exceeds MAX_PAGES

DEPENDENCIES:
- PyPDF2==3.0.1
- pandas==2.0.3
- numpy==1.24.3
"""

import os
import json
from typing import List, Dict
import PyPDF2
import pandas as pd
import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class PDFProcessor:
    """
    Handles PDF file processing and text extraction.
    Ensures consistent output through seed management.
    """
    
    def __init__(self, max_pages: int = 1000):
        """
        Initialize PDF processor with configuration.
        
        Args:
            max_pages: Maximum number of pages to process
        """
        self.max_pages = max_pages  # Hyperparameter: controls processing limit
        self.chunk_size = 512  # Hyperparameter: text chunking size
        
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Process:
        1. Opens PDF file using PyPDF2
        2. Iterates through pages up to max_pages limit
        3. Extracts and concatenates text
        
        Args:
            file_path: Path to input PDF file
            
        Returns:
            Extracted text as string
        """
        # Implementation here...
        pass

print("✅ PDF processing module loaded successfully")
```

#### Docker Files
```dockerfile
# ============================================================================
# FILE: Dockerfile
# LOCATION: 01-data_ingestion/02-pdf_extraction/Dockerfile
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Container for PDF extraction service with Python dependencies
# ============================================================================

# Base image selection - Python 3.10 for compatibility with PyPDF2
FROM python:3.10-slim

# Set working directory for application files
WORKDIR /app

# Copy requirements first for layer caching optimization
COPY requirements.txt .

# Install Python dependencies without cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for consistent execution
ENV PYTHONUNBUFFERED=1
ENV RANDOM_SEED=42

# Execute main processing script
CMD ["python", "main.py"]
```

#### Docker Compose Files
```yaml
# ============================================================================
# FILE: docker-compose.yml
# LOCATION: 01-data_ingestion/02-pdf_extraction/docker-compose.yml
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Orchestrates PDF extraction service with volume mounts
# ============================================================================

version: '3.8'

services:
  pdf_extractor:
    # Build from local Dockerfile
    build: .
    
    # Container naming for easy identification
    container_name: pdf_extraction_service
    
    # Volume mounts for input/output data persistence
    volumes:
      - ./input:/app/input    # Input PDF files
      - ./output:/app/output  # Extracted text output
      - ./logs:/app/logs      # Processing logs
    
    # Environment configuration
    environment:
      - MAX_PAGES=1000        # Hyperparameter: page processing limit
      - CHUNK_SIZE=512        # Hyperparameter: text chunk size
      - LOG_LEVEL=INFO        # Logging verbosity
    
    # Resource limits - no GPU required for this service
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### Step 2: README Creation Template

Create a README.md following this EXACT format:

```markdown
# Pipeline 01-02: PDF Text Extraction

## 🔢 Pipeline Sequence: `01.02`
**Execution Order**: This is sub-pipeline 2 of main pipeline 01 (Data Ingestion)
- **Previous**: 01.01 - File Discovery
- **Current**: 01.02 - PDF Text Extraction
- **Next**: 01.03 - Text Preprocessing

## 📋 Purpose
This pipeline extracts text content from PDF documents for downstream NLP processing. It handles multi-page PDFs, maintains consistent output formatting, and provides error handling for corrupted files.

## ⚙️ Hyperparameters

| Parameter | Default | File | Method/Line | Description |
|-----------|---------|------|-------------|-------------|
| MAX_PAGES | 1000 | data_processor.py | extract_text() L45 | Maximum pages to process per PDF |
| CHUNK_SIZE | 512 | data_processor.py | clean_content() L78 | Text chunking size for processing |
| MIN_TEXT_LENGTH | 100 | data_processor.py | validate_output() L92 | Minimum valid text length |
| BATCH_SIZE | 50 | main.py | process_batch() L34 | Number of PDFs to process in parallel |

## 🎲 Seeds and Reproducibility

| Seed Name | Value | File | Purpose |
|-----------|-------|------|---------|
| RANDOM_SEED | 42 | data_processor.py | Ensures consistent sampling when files exceed MAX_PAGES |
| NP_SEED | 42 | utils.py | NumPy operations reproducibility |

Seeds are set at module initialization to ensure consistent output across runs.

## 🚀 Execution Process

### Build Sequence:
1. Build Docker image: `docker build -t pdf-extractor .`
2. Prepare input/output directories
3. Place PDF files in `./input` directory
4. Run processing pipeline

### Execution Order:
1. **main.py** - Entry point, orchestrates the pipeline
2. **file_discovery.py** - Scans input directory for PDFs
3. **data_processor.py** - Processes each PDF file
4. **output_manager.py** - Handles output formatting and storage
5. **cleanup.py** - Post-processing cleanup (optional)

### Script Execution:
```bash
# Option 1: Direct Python execution
python main.py --input ./input --output ./output

# Option 2: Docker execution (recommended)
docker-compose up
```

## 💻 System Requirements

- **GPU Required**: ❌ No
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB
- **CPU Cores**: 2+ recommended
- **Storage**: 10GB minimum for processing buffer

## 🐳 Docker Execution

### Quick Start:
```bash
# Clone the repository
git clone <repository>
cd 01-data_ingestion/02-pdf_extraction

# Build and run with docker-compose
docker-compose up --build

# Or run in background
docker-compose up -d
```

### Custom Configuration:
```bash
# Override hyperparameters
docker-compose run -e MAX_PAGES=500 -e CHUNK_SIZE=256 pdf_extractor

# Mount custom directories
docker run -v /path/to/pdfs:/app/input -v /path/to/output:/app/output pdf-extractor
```

## 📁 Folder Structure

```
02-pdf_extraction/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── main.py                 # Entry point
├── data_processor.py       # Core processing logic
├── file_discovery.py       # Input file detection
├── output_manager.py       # Output handling
├── utils.py               # Helper functions
├── config.py              # Configuration management
├── scripts/
│   ├── setup.sh          # Environment setup
│   └── validate.sh       # Output validation
├── input/                # Input PDF files (mounted)
├── output/               # Processed output (mounted)
└── logs/                 # Processing logs
```

## 🔧 Configuration

### Environment Variables (.env):
```env
# Processing Configuration
MAX_PAGES=1000
CHUNK_SIZE=512
MIN_TEXT_LENGTH=100
BATCH_SIZE=50

# Paths
INPUT_PATH=/app/input
OUTPUT_PATH=/app/output
LOG_PATH=/app/logs

# Logging
LOG_LEVEL=INFO

# Seeds
RANDOM_SEED=42
```

## 📊 Output Format

Processed data is saved as JSON with the following structure:
```json
{
  "source_file": "document.pdf",
  "processed_date": "2024-01-15T10:30:00Z",
  "pages_processed": 150,
  "text_content": "...",
  "metadata": {
    "total_pages": 200,
    "extraction_time": 45.2,
    "file_size_mb": 12.5
  }
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce BATCH_SIZE or MAX_PAGES |
| Corrupted PDF | Check logs for specific file, will be skipped automatically |
| Slow processing | Increase CPU allocation in docker-compose |
| Missing dependencies | Rebuild Docker image with `--no-cache` flag |

## 📝 Notes

- Processing time varies based on PDF complexity and size
- Encrypted PDFs are not supported and will be logged as errors
- Output is always UTF-8 encoded
- Maintains original paragraph structure where possible

---
✨ Pipeline ready for execution
```

## Processing Order

When analyzing a folder, process files in this order:
1. Configuration files (.env, config.py, settings.json)
2. Main entry points (main.py, app.py, server.py)
3. Core processing modules (in alphabetical order)
4. Utility/helper files
5. Docker files (Dockerfile, docker-compose.yml)
6. Shell scripts
7. Requirements/dependency files
8. Create README.md last (after understanding all components)

## Output Requirements

1. **Comment Consistency**: All comments use the same format and style
2. **No Emojis**: Except for single completion emoji at end of print statements
3. **Comprehensive Examples**: Each complex function should have usage examples in docstrings
4. **Hyperparameter Tracking**: Document every configurable value
5. **Seed Documentation**: Track all random seeds for reproducibility
6. **Clear Dependencies**: List all external libraries with versions

## Example Analysis Output

When you analyze a folder, respond with:

```
ANALYSIS COMPLETE: 01-data_ingestion/02-pdf_extraction

Files Processed:
✅ main.py - Standardized comments, documented 5 methods
✅ data_processor.py - Standardized comments, documented 8 methods, 1 class
✅ Dockerfile - Added comprehensive build comments
✅ docker-compose.yml - Documented all services and configurations
✅ requirements.txt - Added version pins for all dependencies
✅ README.md - Created comprehensive documentation

Hyperparameters Found:
- MAX_PAGES: 1000 (data_processor.py)
- CHUNK_SIZE: 512 (data_processor.py)
- BATCH_SIZE: 50 (main.py)

Seeds Documented:
- RANDOM_SEED: 42 (ensures reproducibility)

File Renames:
- process_data.py → data_processor.py (standard naming convention)

GPU Required: No
Estimated Processing Time: 5-10 minutes for 100 PDFs
Docker Image Size: ~500MB

Ready for pipeline execution ✅
```

## Remember:
- NEVER alter the working logic of the code
- ALWAYS preserve functionality
- ONLY modify comments and documentation
- CREATE comprehensive but clear documentation
- ENSURE all examples are copy-paste ready

## Final Comment Guidelines:

### DO:
- Use everyday analogies (like "this is like sorting mail into different boxes")
- Explain complex concepts in simple terms
- Add context about WHY something is done, not just what
- Use comparisons to real-world activities
- Write as if explaining to a smart person who has never coded
- Add examples of how to use functions
- Explain what each number/parameter means in plain English

### DON'T:
- Use technical jargon without explanation
- Assume knowledge of programming concepts
- Write comments like "instantiate object" or "iterate array"
- Use acronyms without spelling them out first
- Make comments shorter than the code they explain
- Skip commenting because something seems "obvious"

### Comment Length Guide:
- Single line of simple code: 1 line comment
- Complex line or calculation: 2-3 line comment explaining the logic
- Function/Method: 5-10 lines explaining purpose, process, and example
- Class: 10-15 lines explaining the concept, purpose, and how it works
- Complex algorithm: Break down into steps with a comment for each step

### Final Check:
Before finishing, read through all comments and ask:
"Would my non-technical friend understand what this code does and why?"

If the answer is no, rewrite the comments in simpler language.
# Pipeline 01-02: UK Biobank Showcase Data Restructuring

## 🔢 Pipeline Sequence: `01.02`
**Execution Order**: This is sub-pipeline 2 of main pipeline 01 (Data Ingestion)
- **Previous**: 01.01 - UK Biobank Showcase Data Ingestion
- **Current**: 01.02 - UK Biobank Showcase Data Restructuring
- **Next**: 01.03 - PDF Corpus Ingestion

## 📋 Purpose
This pipeline restructures UK Biobank showcase data by linking fields with categories and dictionary metadata, adding intelligent acronym mappings using the Medical Abbreviation and Acronym Meta-Inventory with SPECTER 2 embeddings. It creates comprehensive field-to-dictionary codebooks and validates data integrity.

## ⚙️ Hyperparameters

| Parameter | Default | File | Method/Line | Description |
|-----------|---------|------|-------------|-------------|
| USE_COMPREHENSIVE_TRACKING | True | process_showcase_meta.py | main() L89 | Enables detailed acronym tracking and validation |
| USE_EMBEDDINGS | True | process_showcase_meta.py | main() L90 | Enables SPECTER 2 embedding-based acronym selection |
| MAX_ACRONYMS_PER_TITLE | [1,2,3,4] | process_showcase_meta.py | META_INVENTORY_CFG L95 | Maximum acronyms to replace per field title |
| SIMILARITY_THRESHOLD | 0.65 | process_showcase_meta.py | META_INVENTORY_CFG L135 | Minimum embedding similarity for acronym acceptance |
| HIGH_CONFIDENCE_THRESHOLD | 0.91 | process_showcase_meta.py | META_INVENTORY_CFG L136 | Auto-accept threshold for high-confidence mappings |
| EMBEDDING_BATCH_SIZE | 128 | process_showcase_meta.py | META_INVENTORY_CFG L134 | Batch size for embedding computation |

## 🎲 Seeds and Reproducibility

| Seed Name | Value | File | Purpose |
|-----------|-------|------|---------|
| RANDOM_SEED | Not set | All scripts | Uses default numpy/Python random state |
| NP_SEED | Not set | All scripts | Uses default numpy random state |

No explicit random seeds are set as the processing is deterministic based on input data.

## 🚀 Execution Process

### Build Sequence:
1. Ensure input data from pipeline 01.01 is available
2. Build Docker image with GPU support: `docker build -t ukb-restructure .`
3. Prepare output directories
4. Run processing pipeline with appropriate service

### Execution Order:
1. **process_showcase_meta.py** - Main Meta-Inventory processing with SPECTER 2 embeddings
2. **field_category_check.py** - Data integrity validation
3. **field_check.py** - Category summary generation
4. **ai_consortium_validation.py** - AI-powered acronym validation (optional)

### Script Execution:
```bash
# Primary: Meta-Inventory with embeddings (GPU recommended)
docker compose up ukb-meta-processor-gpu

# Alternative: CPU-only Meta-Inventory processing
docker compose up ukb-meta-processor

# Validation and reporting
docker compose up field-category-check
docker compose up field-summary

# AI validation (requires OPENROUTER_API_KEY)
docker compose up ai-consortium
```

## 💻 System Requirements

- **GPU Required**: Optional (recommended for Meta-Inventory processing with embeddings)
- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB+
- **CPU Cores**: 4+ recommended
- **Storage**: 50GB minimum (for embeddings cache and output)
- **CUDA**: 12.8+ (if using GPU)

## 🐳 Docker Execution

### Quick Start:
```bash
# Clone the repository
git clone <repository>
cd pipelines/01-data_ingestion/02-restructure_ukb_showcase_data

# Build and run with Meta-Inventory processing (GPU)
docker compose up --build ukb-meta-processor-gpu

# Or run CPU-only Meta-Inventory processing
docker compose up --build ukb-meta-processor

# View processing logs
docker compose logs -f ukb-meta-processor-gpu
```

### Custom Configuration:
```bash
# Override embedding similarity threshold
docker compose run -e SIMILARITY_THRESHOLD=0.7 ukb-meta-processor-gpu

# Disable embeddings for CPU-only processing
docker compose run -e USE_EMBEDDINGS=False ukb-meta-processor

# Run AI validation (requires OPENROUTER_API_KEY)
docker compose run ai-consortium
```

## 📁 Folder Structure

```
02-restructure_ukb_showcase_data/
├── Dockerfile                    # GPU-enabled container with SPECTER 2
├── docker-compose.yaml          # Multi-service orchestration
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── scripts/
│   ├── process_showcase_meta.py # Meta-Inventory processing with SPECTER 2 embeddings
│   ├── field_check.py           # Category summary generation
│   ├── field_category_check.py  # Data integrity validation
│   └── ai_consortium_validation.py # AI-powered validation
├── output-meta/                 # Processing outputs
│   ├── ukb_fields.*             # Enriched field data
│   ├── ukb_field_codebook.*     # Field-to-dictionary mappings
│   ├── acronym_candidates.*     # Acronym mapping details
│   ├── validation_report.txt    # Processing statistics
│   ├── human_verify.txt         # Manual verification samples
│   ├── acronym_mappings.txt     # Unique mappings summary
│   ├── acronym_summary.txt      # Frequency analysis
│   └── embedding_cache/         # SPECTER 2 cache (large)
└── ../01-ingest_ukb_showcase_data/input/  # Input data (mounted)
```

## 🔧 Configuration

### Environment Variables (.env):
```env
# Processing Configuration
USE_COMPREHENSIVE_TRACKING=True
USE_EMBEDDINGS=True

# Embedding Configuration
EMBEDDING_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
EMBEDDING_BATCH_SIZE=128
SIMILARITY_THRESHOLD=0.65
HIGH_CONFIDENCE_THRESHOLD=0.91

# API Keys (for AI validation)
OPENROUTER_API_KEY=your_api_key_here

# Paths
INPUT_PATH=/input
OUTPUT_PATH=/output
```

## 📊 Output Format

### Primary Outputs:
- **ukb_fields.csv/parquet**: Enriched field data with categories and acronyms
- **ukb_field_codebook.csv/parquet**: Complete field-to-dictionary code mappings
- **acronym_candidates.csv/parquet**: Detailed acronym selection tracking

### Validation Outputs:
- **validation_report.txt**: Processing statistics and data quality metrics
- **human_verify.txt**: Samples for manual verification
- **field_category_title_matches.csv**: Data integrity check results

### Analysis Outputs:
- **acronym_mappings.txt**: Unique acronym mappings
- **acronym_summary.txt**: Acronym usage frequency analysis
- **category_summary.csv**: Fields per category statistics

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce EMBEDDING_BATCH_SIZE or disable USE_EMBEDDINGS |
| Slow processing | Use GPU-enabled service (ukb-meta-processor-gpu) |
| Missing Meta-Inventory file | Ensure meta-inventory CSV is in input directory |
| Embedding cache too large | Set CACHE_EMBEDDINGS=False in environment |
| API rate limits | AI validation service may hit OpenRouter limits |

## 📝 Notes

- Processing time varies based on input size and embedding usage (5-30 minutes)
- GPU acceleration strongly recommended for Meta-Inventory processing with SPECTER 2
- Meta-Inventory provides high-quality acronym mappings using semantic similarity
- Output files can be large (>100MB) depending on UK Biobank data size
- Embedding cache persists between runs for efficiency
- Multi-column acronym generation creates all possible combinations for flexibility

---
✨ Pipeline ready for execution
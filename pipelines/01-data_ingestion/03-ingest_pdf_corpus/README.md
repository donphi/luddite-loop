# Pipeline 01-03: PDF Corpus Ingestion

## 🔢 Pipeline Sequence: `01.03`
**Execution Order**: This is sub-pipeline 3 of main pipeline 01 (Data Ingestion)
- **Previous**: 01.02 - PDF Text Extraction
- **Current**: 01.03 - PDF Corpus Ingestion
- **Next**: 02.01 - Data Conversion

## 📋 Purpose
This pipeline downloads PDF documents for academic publications from multiple open access sources. It implements a multi-stage approach to maximize coverage by trying different academic repositories and APIs in sequence.

## ⚙️ Hyperparameters

| Parameter | Default | File | Method/Line | Description |
|-----------|---------|------|-------------|-------------|
| REQUEST_TIMEOUT | 10-30s | Various | download_pdf() | HTTP request timeout for PDF downloads |
| MIN_FILE_SIZE | 1000 bytes | 04-validate_dedupe.py | validate_pdf() L107 | Minimum valid file size |
| MIN_PAGE_COUNT | 2 | 04-validate_dedupe.py | validate_pdf() L128 | Minimum pages for valid PDF |
| MAX_FILENAME_LENGTH | 200 chars | All fetchers | create_filename() | Maximum filename length |
| SIMILARITY_THRESHOLD | 0.7-0.8 | Various | calculate_similarity() | Title matching threshold |

## 🎲 Seeds and Reproducibility

| Seed Name | Value | File | Purpose |
|-----------|-------|------|---------|
| N/A | N/A | N/A | No random operations in this pipeline |

Seeds are not used in this pipeline as all operations are deterministic.

## 🚀 Execution Process

### Build Sequence:
1. Configure environment variables in `.env`
2. Prepare publications file in TSV format
3. Build Docker image: `docker compose build`
4. Run pipeline stages sequentially

### Execution Order:
1. **01-openalex_fetcher.py** - Downloads from OpenAlex API (primary source)
2. **02-core_fetcher.py** - Downloads from CORE API (secondary source)
3. **03-other_fetcher.py** - Downloads from Unpaywall, PMC, arXiv (additional sources)
4. **04-validate_dedupe.py** - Validates PDFs and removes duplicates

### Script Execution:
```bash
# Option 1: Run all stages sequentially
docker compose --profile full up

# Option 2: Run individual stages
docker compose --profile stage1 up  # OpenAlex
docker compose --profile stage2 up  # CORE
docker compose --profile stage3 up  # Other sources
docker compose --profile stage4 up  # Validation
```

## 💻 System Requirements

- **GPU Required**: No
- **Minimum RAM**: 2GB
- **Recommended RAM**: 4GB
- **CPU Cores**: 1+ recommended
- **Storage**: 50GB minimum for PDF storage

## 🐳 Docker Execution

### Quick Start:
```bash
# Clone the repository
git clone <repository>
cd 01-data_ingestion/03-ingest_pdf_corpus

# Configure environment
cp .env.example .env
# Edit .env with your API tokens and file paths

# Run the full pipeline
docker-compose --profile full up
```

### Custom Configuration:
```bash
# Override environment variables
docker-compose run -e EMAIL=your.email@domain.com openalex-fetcher

# Mount custom publications file
docker run -v /path/to/publications.txt:/app/data/publications.txt pdf-corpus
```

## 📁 Folder Structure

```
03-ingest_pdf_corpus/
├── .env.example              # Environment template
├── .env                      # Environment configuration
├── docker-compose.yaml       # Multi-stage orchestration
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
├── scripts/
│   ├── 01-openalex_fetcher.py # OpenAlex API downloader
│   ├── 02-core_fetcher.py     # CORE API downloader
│   ├── 03-other_fetcher.py    # Additional sources downloader
│   └── 04-validate_dedupe.py  # PDF validation and deduplication
└── output/                  # Generated output directory
    ├── openalex/            # Stage 1 raw downloads
    ├── core/                # Stage 2 raw downloads
    ├── other/               # Stage 3 raw downloads
    ├── valid_pdfs/          # Final validated PDFs
    ├── invalid_pdfs/        # Corrupted PDFs
    ├── duplicate_pdfs/      # Removed duplicates
    ├── publications/        # Tracking files
    └── validation_report.txt
```

## 🔧 Configuration

### Environment Variables (.env):
```env
# Required: Path to your publications file
PUBLICATIONS_FILE=/app/data/publications.txt

# Required: Your email address (for API requests)
EMAIL=your.email@university.edu

# Optional but recommended: API tokens
OPENALEX_TOKEN=your_openalex_token_here
CORE_API_KEY=your_core_api_key_here
```

### Publications File Format:
Tab-separated values with required columns:
- `pub_id`: Unique publication identifier
- `title`: Publication title
- `doi`: DOI (optional but recommended)
- `year_pub`: Publication year

## 📊 Output Format

### Valid PDFs (`output/valid_pdfs/`):
- One PDF per unique publication
- Standardized filenames: `{doi}_{title_words}_{year}.pdf`
- Validated for readability and content

### Tracking Files (`output/publications/`):
- `publications_openalex.txt`: OpenAlex download status
- `publications_core.txt`: CORE download status
- `publications_other.txt`: Other sources download status

### Validation Report (`output/validation_report.txt`):
```
PDF Validation and Deduplication Report
====================================

SUMMARY:
Total PDFs: 1500
Valid PDFs: 1200
Invalid PDFs: 200
Duplicates Removed: 100
Final Unique PDFs: 1100
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No PDFs found | Check publications file format (tab-separated) |
| API rate limits | Add API tokens, increase delays between requests |
| Permission denied | Check Docker volume mounts and file permissions |
| SSL errors | Pipeline handles most SSL issues automatically |
| Low success rate | Check institutional VPN access for paywalled content |

## 📝 Notes

- Success rates vary by publication age and subject area
- Recent publications may not be indexed in all sources yet
- Institutional access can significantly improve success rates
- Pipeline is designed for batch processing of large publication lists

---
✨ Pipeline ready for PDF corpus ingestion
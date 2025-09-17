# Academic PDF Download Pipeline

A simple, reproducible 4-stage pipeline for downloading academic PDFs from multiple sources using Docker.

## Overview

This pipeline downloads PDFs for academic publications through 4 sequential stages:

1. **OpenAlex** - Downloads from OpenAlex API (primary source)
2. **CORE** - Downloads from CORE API (secondary source)  
3. **Other Sources** - Downloads from Unpaywall, PMC, arXiv (additional sources)
4. **Validation & Deduplication** - Validates PDFs and removes duplicates

## Features

- ✅ **Consistent file naming** across all sources
- ✅ **Automatic deduplication** (keeps best quality PDF)
- ✅ **PDF validation** (removes broken/corrupted files)
- ✅ **Docker containerized** for reproducibility
- ✅ **Simple stage-by-stage execution**
- ✅ **Comprehensive logging and reporting**

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **API tokens** (see Configuration section)
3. **Publications file** in tab-separated format

## Quick Start

1. **Clone/download this repository**
2. **Create environment file** (see Configuration)
3. **Prepare your publications file**
4. **Run the pipeline**

```bash
# Run all stages sequentially
docker compose --profile full up

# OR run individual stages
docker compose --profile stage1 up  # OpenAlex
docker compose --profile stage2 up  # CORE  
docker compose --profile stage3 up  # Other sources
docker compose --profile stage4 up  # Validation
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```env
# Required: Path to your publications file
PUBLICATIONS_FILE=/path/to/your/publications.txt

# Required: Your email address (for API requests)
EMAIL=your.email@university.edu

# Optional but recommended: API tokens
OPENALEX_TOKEN=your_openalex_token_here
CORE_API_KEY=your_core_api_key_here
```

### 2. Get API Tokens

#### OpenAlex Token (Recommended)
- Go to [OpenAlex](https://openalex.org/)
- Sign up for free API access
- Get your token from account settings
- Higher rate limits with token

#### CORE API Key (Recommended)  
- Go to [CORE API](https://core.ac.uk/services/api/)
- Register for free API key
- Required for full CORE access

### 3. Publications File Format

Your publications file must be **tab-separated** with these columns:

```
pub_id	title	doi	year_pub	journal	authors
pub_001	Sample Paper Title	10.1000/sample	2023	Nature	Smith, J.
pub_002	Another Study	10.1000/another	2022	Science	Jones, A.
```

**Required columns:**
- `pub_id` - Unique identifier for each publication
- `title` - Publication title
- `doi` - DOI (if available, otherwise can be empty)
- `year_pub` - Publication year

**Optional columns:**
- `journal`, `authors`, etc. (for reference only)

## File Naming Convention

All PDFs use a consistent naming format:

```
{normalized_doi}_{first_5_title_words}_{year}.pdf
```

**Examples:**
- `10_1038_nature12345_machine_learning_protein_folding_2023.pdf`
- `no_doi_covid_vaccine_efficacy_study_2022.pdf`

This ensures:
- ✅ Same paper from different sources gets same filename
- ✅ Easy deduplication based on filename
- ✅ Human-readable file identification

## Directory Structure

```
project/
├── 1_openalex_fetcher.py
├── 2_core_fetcher.py
├── 3_other_fetcher.py
├── 4_validate_and_dedupe.py
├── docker compose.yml
├── Dockerfile
├── requirements.txt
├── .env
└── output/
    ├── openalex/          # Stage 1 downloads
    ├── core/              # Stage 2 downloads
    ├── other/             # Stage 3 downloads
    ├── valid_pdfs/        # Final valid PDFs
    ├── invalid_pdfs/      # Broken/corrupted PDFs
    ├── duplicate_pdfs/    # Removed duplicates
    └── validation_report.txt
```

## Usage Instructions

### Option 1: Run Full Pipeline

```bash
# Run all 4 stages automatically
docker compose --profile full up
```

This runs all stages sequentially. Best for unattended operation.

### Option 2: Run Individual Stages

```bash
# Stage 1: OpenAlex
docker compose --profile stage1 up

# Stage 2: CORE
docker compose --profile stage2 up

# Stage 3: Other sources
docker compose --profile stage3 up

# Stage 4: Validation & deduplication
docker compose --profile stage4 up
```

Run stages individually for more control and monitoring.

### Option 3: Development/Testing

```bash
# Build the container
docker compose build

# Run specific script manually
docker run -v ./output:/app/output -v /path/to/publications.txt:/app/data/publications.txt pdf-pipeline python 1_openalex_fetcher.py
```

## Output Explanation

After completion, check these directories:

### `output/valid_pdfs/` - YOUR FINAL PDFS ✅
- Contains validated, deduplicated PDFs
- These are ready for analysis/processing
- One PDF per unique publication

### `output/invalid_pdfs/` - Problematic files ❌
- Corrupted PDFs
- Encrypted PDFs  
- Files too small (<2 pages)
- Cannot be used

### `output/duplicate_pdfs/` - Removed duplicates 📋
- Lower quality versions of papers
- Multiple downloads of same paper
- Kept the version with most pages

### Individual stage directories:
- `output/openalex/` - Raw downloads from OpenAlex
- `output/core/` - Raw downloads from CORE  
- `output/other/` - Raw downloads from other sources

## Monitoring Progress

Each stage provides detailed logging:

```bash
# View logs for specific stage
docker compose --profile stage1 up --logs

# Follow logs in real-time
docker compose --profile stage1 up --follow
```

**Log messages to watch for:**
- `✓ Downloaded: filename.pdf` - Successful download
- `Not found in [source]` - Paper not available in that source
- `File already exists` - Skipping duplicate download
- `Moved to invalid` - PDF failed validation

## Troubleshooting

### Common Issues

**1. "No PDFs found"**
- Check your publications file format (tab-separated)
- Verify `pub_id`, `title`, `doi` columns exist
- Ensure DOIs are properly formatted

**2. "API rate limit exceeded"**
- Add delays between requests (modify `time.sleep()` values)
- Get API tokens for higher rate limits
- Run stages separately with breaks

**3. "Permission denied"**
- Check Docker has access to your publications file
- Verify output directory permissions
- Try running with `sudo` if needed

**4. "SSL/Certificate errors"**
- Some academic sites have SSL issues
- Pipeline automatically handles most cases
- Check network connectivity

### Performance Tips

- **Use API tokens** - Much higher rate limits
- **Run overnight** - Academic sites are faster off-peak
- **Monitor disk space** - PDFs can be large
- **Check institution access** - Some sources need university VPN

## Expected Results

**Typical success rates:**
- OpenAlex: 60-80% (depends on DOI availability)
- CORE: 40-60% (open access papers)
- Other sources: 20-40% (additional coverage)
- **Combined: 70-90%** coverage for recent papers

**Final output:**
- 1 validated PDF per unique publication
- Consistent filenames for easy processing
- Detailed report of success/failure rates

## Citation & Credits

This pipeline uses APIs from:
- [OpenAlex](https://openalex.org/) - Open bibliographic database
- [CORE](https://core.ac.uk/) - Open access research aggregator  
- [Unpaywall](https://unpaywall.org/) - Open access finder
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) - Life sciences repository

## License

MIT License - Feel free to modify and distribute.

## Support

For issues:
1. Check the troubleshooting section
2. Review log files for error details
3. Verify your input file format
4. Test with a small subset first

---

## ⚠️ Important Data Availability Limitations

Before proceeding, be aware of these critical limitations:

- **Publication Lag**: New UK Biobank publications take time to be indexed in OpenAlex, CORE, and other academic databases. Recent publications may not have PDFs available yet.
- **Incomplete Corpus**: You will **not** be able to download the entire publication corpus due to various access restrictions.
- **Paywall Restrictions**: Many publications are behind publisher paywalls, limiting full-text PDF access even when metadata is available.
- **Best Effort Extraction**: Expect to retrieve approximately 40-70% of the full corpus depending on your institutional access and the age of publications.

---



**Happy PDF hunting! 📚**
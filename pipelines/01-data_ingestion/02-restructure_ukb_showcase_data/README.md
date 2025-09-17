## Getting Started

## Running the Scripts

### Prerequisites
Place all downloaded files (`field.txt`, `category.txt`, `publications.txt`) in the `01-ingest_ukb_shocase_data/input` folder

### Docker Commands
```bash
# Run all processing scripts in sequence
docker compose up

# Or run individual services:
# 1. Main processor (run this first)
docker compose up ukb-processor

# 2. Data validation check
docker compose up field-category-check

# 3. Category summary
docker compose up field-summary

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
#!/bin/bash
# Run manual download helper on host machine
# This script runs outside Docker to access your browser

# Ensure output directories exist
mkdir -p ./output/manual
mkdir -p ./output/manual_renamed

# Check if required Python packages are installed
python3 -c "import PyPDF2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required Python packages..."
    pip3 install PyPDF2 requests
fi

# Run the helper directly
python3 04-manual_download_helper.py

echo ""
echo "Manual downloads complete. Re-run validation with:"
echo "docker compose --profile stage4 up"
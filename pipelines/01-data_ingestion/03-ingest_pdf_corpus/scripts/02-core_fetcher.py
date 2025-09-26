# ============================================================================
# FILE: scripts/02-core_fetcher.py
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/scripts/02-core_fetcher.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Fetches PDF documents from CORE (COnnecting REpositories) API
# ============================================================================

"""
MODULE OVERVIEW:
This module downloads PDF documents from the CORE academic repository API.
CORE provides access to millions of open access research papers. This fetcher
includes robust retry logic for handling rate limits and server errors.

CLASSES:
- CoreFetcher: Main class for handling CORE API interactions and PDF downloads

METHODS:
- __init__(): Initializes the fetcher with environment variables and creates output directories
- init_tracking_file(): Creates a tracking file to record download status for each publication
- update_tracking_file(): Updates the tracking file with success/failure status
- normalize_text(): Cleans text for consistent filename generation
- create_filename(): Generates standardized PDF filenames from publication metadata
- load_publications(): Reads publication data from input TSV file
- make_api_request_with_retry(): Makes API calls with exponential backoff for rate limits
- search_core(): Searches CORE for a publication using DOI or title
- search_by_doi(): Searches CORE specifically by DOI identifier
- search_by_title(): Searches CORE by title and verifies publication year
- calculate_similarity(): Computes text similarity for title matching
- get_pdf_urls(): Extracts PDF download URLs from CORE work data
- download_pdf(): Downloads PDF from URL with content validation
- process_publications(): Main orchestration method that processes all publications
- print_summary(): Displays final statistics of the download operation

ROUTES:
- N/A (This is a data processing module, not a web service)

HYPERPARAMETERS:
- REQUEST_TIMEOUT: 15 seconds (in make_api_request_with_retry method)
- DOWNLOAD_TIMEOUT: 30 seconds (in download_pdf method)
- MAX_RETRIES: 5 (in make_api_request_with_retry method)
- MIN_FILE_SIZE: 1000 bytes (in download_pdf method, minimum valid file size)
- SIMILARITY_THRESHOLD: 0.8 (in search_by_title method, for title matching)
- MAX_FILENAME_LENGTH: 200 characters (in create_filename method)
- RATE_LIMIT_DELAY: 12 seconds (in process_publications method, between publications)

SEEDS:
- N/A (No random seeds used in this module)

DEPENDENCIES:
- requests: For HTTP API calls and file downloads
- pathlib: For cross-platform file path handling
- csv: For reading tab-separated publication data
- logging: For operation tracking and debugging
- re: For text normalization and pattern matching
- time: For rate limiting between API calls
"""

import os
import csv
import requests
import time
import logging
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreFetcher:
    """
    Main class for fetching PDFs from CORE (COnnecting REpositories) API.

    This class handles the complete pipeline of:
    1. Loading publication metadata from a file
    2. Searching CORE for each publication with retry logic
    3. Finding available PDF download links
    4. Downloading PDFs with validation
    5. Tracking success/failure for each publication
    """

    def __init__(self):
        """
        Initialize the CORE fetcher with configuration from environment variables.

        Think of this like preparing to visit a very strict, high-security library.
        CORE (COnnecting REpositories) is like a librarian that manages access to
        millions of research papers, but they're very particular about how you ask
        for things and how often you can ask.

        Unlike a casual bookstore where you can browse freely, CORE requires:
        - Proper identification (API key)
        - Polite, structured requests
        - Patience between requests (rate limiting)
        """
        # Get file paths from environment - our shopping list of papers to find
        self.input_file = os.getenv("PUBLICATIONS_FILE", "/app/data/publications.txt")
        self.output_dir = Path("/app/output/core")
        self.core_api_key = os.getenv('CORE_API_KEY', '')

        # Create the folder where we'll save downloaded PDFs
        # Like organizing your briefcase before going to the library
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Set up HTTP headers for CORE API
        # CORE requires Bearer token authentication - like showing your library card
        # The "Bearer" part is like saying "I'm authorized to be here"
        self.base_url = "https://api.core.ac.uk/v3"
        self.headers = {
            'Authorization': f'Bearer {self.core_api_key}',
            'Content-Type': 'application/json'
        }

        # Warn if no API key - like trying to enter a library without membership
        # CORE allows some requests without a key but severely limits how many
        if not self.core_api_key:
            logger.warning("No CORE API key provided - requests may be rate limited")
        else:
            logger.info("Using CORE API key for higher rate limits")

        # Track how many publications we process, find, download, etc.
        # Like keeping a scorecard to see how successful our library visit is
        self.stats = {'processed': 0, 'found': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

    def init_tracking_file(self, publications):
        """Initialize the tracking file with header from original file"""
        self.tracking_file = Path(self.output_dir) / "publications" / "publications_core.txt"
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy header from original file
        with open(self.input_file, 'r', encoding='utf-8') as f:
            header = f.readline()
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            f.write(header)
        
        logger.info(f"Initialized tracking file: {self.tracking_file}")

    def update_tracking_file(self, pub_data, status, error_msg=None):
        """Update tracking file with download status"""
        # Get the original line
        original_line = '\t'.join([pub_data.get(key, '') for key in pub_data.keys()])
        
        if status == 'success':
            # Just copy the line as-is for successful downloads
            line_to_write = original_line
        else:
            # Append error message for failed downloads
            line_to_write = f"{original_line}\t[Error - Not downloaded - {error_msg}]"
        
        with open(self.tracking_file, 'a', encoding='utf-8') as f:
            f.write(line_to_write + '\n')

    def normalize_text(self, text):
        """Normalize text for consistent naming (same as OpenAlex)"""
        if not text:
            return ""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'\s+', '_', text.strip())
        return text

    def create_filename(self, pub_data, core_id):
        """Create consistent filename format: {doi}_{title_words}_{year}.pdf"""
        # Normalize DOI
        doi = pub_data.get('doi', '').lower()
        doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        doi = doi.replace('doi:', '').strip('/')
        doi = self.normalize_text(doi) if doi else 'no_doi'
        
        # Get first 5 words of title
        title = pub_data.get('title', '')
        title_words = self.normalize_text(title).split('_')[:5]
        title_part = '_'.join(title_words) if title_words else 'no_title'
        
        # Get year
        year = pub_data.get('year_pub', '').strip() or 'no_year'
        
        # Create filename
        filename = f"{doi}_{title_part}_{year}.pdf"
        # Ensure filename isn't too long
        if len(filename) > 200:
            filename = filename[:200] + ".pdf"
        
        return filename

    def load_publications(self):
        """Load publications from input file"""
        publications = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    publications.append(row)
            logger.info(f"Loaded {len(publications)} publications")
            return publications
        except Exception as e:
            logger.error(f"Error loading publications: {e}")
            return []

    def make_api_request_with_retry(self, url, params, max_retries=5):
        """
        Make API request with smart retry logic for rate limits and server errors.
        
        Think of this like dealing with a busy restaurant. Sometimes when you call,
        they say "we're too busy right now, call back in 5 minutes" (rate limit).
        Sometimes their phone system is broken (server error). This method handles
        both situations by waiting and trying again, but not forever.
        
        We use "exponential backoff" - like if they say they're busy, we wait 5 minutes.
        If they're still busy, we wait 10 minutes. Then 20 minutes. We get more patient
        each time, but eventually give up if it's never going to work.
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - the server is saying "slow down, you're asking too fast"
                    # Like a librarian saying "please wait, I'm helping other people too"
                    wait_time = min(60, (2 ** attempt) * 5)  # 5, 10, 20, max 60 seconds
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:
                    # Server errors - something is broken on their end
                    # Like the library's computer system crashing
                    wait_time = min(30, (2 ** attempt) * 2)  # 2, 4, 8 seconds
                    logger.warning(f"Server error ({e.response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    # Other HTTP errors (4xx) - don't retry
                    # Like being told "that book doesn't exist" - no point asking again
                    raise
            except Exception as e:
                # Network errors, timeouts etc - connection problems
                logger.warning(f"Request failed: {e}")
                return None
        
        # We tried our best but it's not working
        logger.warning(f"Failed after {max_retries} retries")
        return None

    def search_core(self, pub_data):
        """Search CORE for publication"""
        # Try DOI first
        doi = pub_data.get('doi', '').strip()
        if doi:
            result = self.search_by_doi(doi)
            if result:
                return result
        
        # Try title if no DOI or DOI failed
        title = pub_data.get('title', '').strip()
        year = pub_data.get('year_pub', '').strip()
        if title:
            return self.search_by_title(title, year)
        
        return None

    def search_by_doi(self, doi):
        """Search CORE by DOI with retry logic"""
        try:
            # Clean DOI for search
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            clean_doi = clean_doi.replace('doi:', '').strip('/')
            
            # Search endpoint
            url = f"{self.base_url}/search/works"
            params = {
                'q': f'doi:"{clean_doi}"',
                'limit': 10
            }
            
            response = self.make_api_request_with_retry(url, params)
            if not response:
                return None
            
            data = response.json()
            results = data.get('results', [])
            
            # Find exact DOI match
            for work in results:
                work_doi = work.get('doi', '').lower()
                if clean_doi.lower() in work_doi or work_doi in clean_doi.lower():
                    return work
            
            return results[0] if results else None
            
        except Exception as e:
            logger.warning(f"CORE DOI search failed: {e}")
            return None

    def search_by_title(self, title, year=None):
        """Search CORE by title with retry logic"""
        try:
            url = f"{self.base_url}/search/works"
            params = {
                'q': title,
                'limit': 10
            }
            
            response = self.make_api_request_with_retry(url, params)
            if not response:
                return None
            
            data = response.json()
            results = data.get('results', [])
            
            # Find best title match
            for work in results:
                work_title = work.get('title', '').lower()
                work_year = str(work.get('publishedDate', '')[:4]) if work.get('publishedDate') else ''
                
                # Check title similarity
                title_similarity = self.calculate_similarity(title.lower(), work_title)
                year_match = not year or work_year == year
                
                if title_similarity > 0.8 and year_match:
                    return work
            
            # If no good match, return first result if exists
            return results[0] if results else None
            
        except Exception as e:
            logger.warning(f"CORE title search failed: {e}")
            return None

    def calculate_similarity(self, text1, text2):
        """Simple word-based similarity"""
        words1 = set(self.normalize_text(text1).split('_'))
        words2 = set(self.normalize_text(text2).split('_'))
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0

    def get_pdf_urls(self, work):
        """
        Extract PDF URLs from CORE work data.
        
        CORE is simpler than OpenAlex - it usually gives us just a few clear options
        for where to find the PDF. Think of it like a librarian who has already
        done the hard work of finding the best copies and just hands you a short
        list of exactly where to look.
        
        We still rank the options by how likely they are to work:
        - fullTextIdentifier: "Here's the official full text" (best option)
        - downloadUrl: "Here's a direct download link" (good option)
        - PDF links: "These other links might have PDFs" (worth trying)
        """
        pdf_urls = []
        
        # Check for fullText download URL - this is CORE's "best bet"
        # Like the librarian saying "this is definitely the right copy"
        if work.get('fullTextIdentifier'):
            pdf_urls.append({
                'url': work['fullTextIdentifier'],
                'source': 'core_fulltext',
                'priority': 90
            })
        
        # Check for download URL - direct download link
        # Like being told "click here to download immediately"
        if work.get('downloadUrl'):
            pdf_urls.append({
                'url': work['downloadUrl'],
                'source': 'core_download',
                'priority': 85
            })
        
        # Check other links that might contain PDFs
        # Like checking "these other places might also have copies"
        for link in work.get('links', []):
            if link.get('url', '').endswith('.pdf'):
                pdf_urls.append({
                    'url': link['url'],
                    'source': 'core_link',
                    'priority': 80
                })
        
        # Sort by priority - try the best options first
        pdf_urls.sort(key=lambda x: x['priority'], reverse=True)
        return pdf_urls

    def download_pdf(self, pdf_info, filename):
        """Download PDF from URL"""
        url = pdf_info['url']
        filepath = os.path.join(self.output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            logger.info(f"  File already exists: {filename}")
            self.stats['skipped'] += 1
            return True
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; COREPDFDownloader/1.0)',
                'Accept': 'application/pdf,*/*'
            }
            
            # Add API key for CORE URLs
            if 'core.ac.uk' in url and self.core_api_key:
                headers['Authorization'] = f'Bearer {self.core_api_key}'
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                logger.warning(f"  Got HTML instead of PDF from {pdf_info['source']}")
                
                # Save URL to manual download list
                manual_file = Path("/app/output/manual_download.txt")
                with open(manual_file, 'a', encoding='utf-8') as f:
                    # Get publication info for context
                    pub_id = self.current_pub_data.get('pub_id', 'unknown')
                    title = self.current_pub_data.get('title', 'unknown')[:80]
                    doi = self.current_pub_data.get('doi', 'no_doi')
                    
                    f.write(f"{pub_id}\t{doi}\t{title}\t{url}\t{pdf_info['source']}\n")
                
                return False
            
            # Download and save
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB is suspicious
                os.remove(filepath)
                logger.warning(f"  Downloaded file too small ({file_size} bytes), removed")
                return False
            
            logger.info(f"  Downloaded: {filename} ({file_size:,} bytes) from {pdf_info['source']}")
            self.stats['downloaded'] += 1
            return True
            
        except Exception as e:
            logger.warning(f"  Download failed from {pdf_info['source']}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def process_publications(self):
        """Main processing function"""
        logger.info("Starting CORE PDF download")
        logger.info("Using Academic rate limit: 10 requests/minute")

        # Create header for manual download file
        manual_file = Path("/app/output/manual_download.txt")
        if not manual_file.exists():
            with open(manual_file, 'w', encoding='utf-8') as f:
                f.write("pub_id\tdoi\ttitle\turl\tsource\n")
        
        publications = self.load_publications()
        if not publications:
            return
        
        # Initialize tracking file
        self.init_tracking_file(publications)
        
        for i, pub_data in enumerate(publications, 1):
            pub_id = pub_data.get('pub_id', f'pub_{i}')
            logger.info(f"Processing {i}/{len(publications)}: {pub_id}")
            
            self.stats['processed'] += 1
            self.current_pub_data = pub_data  # Store for use in download_pdf
            
            # Search CORE
            work = self.search_core(pub_data)
            if not work:
                logger.info(f"  Not found in CORE")
                self.update_tracking_file(pub_data, 'failed', 'Not found in CORE')
                continue
            
            self.stats['found'] += 1
            core_id = work.get('id', 'unknown')
            
            # Get PDF URLs
            pdf_urls = self.get_pdf_urls(work)
            if not pdf_urls:
                logger.info(f"  No PDF URLs found")
                self.update_tracking_file(pub_data, 'failed', 'No PDF URLs available')
                continue
            
            # Create filename (same format as OpenAlex)
            filename = self.create_filename(pub_data, core_id)
            
            # Check if already exists
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"  File already exists: {filename}")
                self.stats['skipped'] += 1
                self.update_tracking_file(pub_data, 'success')
                continue
            
            # Try to download from URLs (in priority order)
            downloaded = False
            last_error = "Download failed from all sources"
            for pdf_info in pdf_urls:
                if self.download_pdf(pdf_info, filename):
                    downloaded = True
                    self.update_tracking_file(pub_data, 'success')
                    break
                time.sleep(0.5)  # Small delay between attempts
            
            if not downloaded:
                logger.info(f"  Failed to download from all sources")
                self.stats['failed'] += 1
                self.update_tracking_file(pub_data, 'failed', last_error)
            
            # Rate limiting for CORE API - Academic tier: 10 per minute
            # Each publication makes 2 API calls, so we need 12 seconds between publications
            time.sleep(12.0)
            
            # Progress update
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(publications)} - "
                        f"Found: {self.stats['found']}, Downloaded: {self.stats['downloaded']}")

    def print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*50)
        logger.info("CORE FETCHER SUMMARY")
        logger.info("="*50)
        for key, value in self.stats.items():
            logger.info(f"{key.capitalize()}: {value:,}")
        
        if self.stats['found'] > 0:
            success_rate = (self.stats['downloaded'] / self.stats['found']) * 100
            logger.info(f"Download Success Rate: {success_rate:.1f}%")

def main():
    fetcher = CoreFetcher()
    fetcher.process_publications()
    fetcher.print_summary()

if __name__ == "__main__":
    main()

print("✅ CORE PDF fetcher module loaded successfully")
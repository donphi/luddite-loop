# ============================================================================
# FILE: scripts/01-openalex_fetcher.py
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/scripts/01-openalex_fetcher.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Fetches PDF documents from OpenAlex API using publication metadata
# ============================================================================

"""
MODULE OVERVIEW:
This module downloads PDF documents from the OpenAlex academic database API.
It searches for publications by DOI or title, then attempts to download PDFs
from multiple sources in order of reliability.

CLASSES:
- OpenAlexFetcher: Main class for handling OpenAlex API interactions and PDF downloads

METHODS:
- __init__(): Initializes the fetcher with environment variables and creates output directories
- init_tracking_file(): Creates a tracking file to record download status for each publication
- update_tracking_file(): Updates the tracking file with success/failure status
- normalize_text(): Cleans text for consistent filename generation
- create_filename(): Generates standardized PDF filenames from publication metadata
- load_publications(): Reads publication data from input TSV file
- search_openalex(): Searches OpenAlex for a publication using DOI or title
- search_by_doi(): Searches OpenAlex specifically by DOI identifier
- search_by_title(): Searches OpenAlex by title and verifies publication year
- calculate_similarity(): Computes text similarity for title matching
- get_pdf_urls(): Extracts and prioritizes PDF download URLs from OpenAlex work data
- get_source_priority(): Assigns reliability scores to different academic sources
- download_pdf(): Downloads PDF from URL with content validation
- process_publications(): Main orchestration method that processes all publications
- print_summary(): Displays final statistics of the download operation

ROUTES:
- N/A (This is a data processing module, not a web service)

HYPERPARAMETERS:
- REQUEST_TIMEOUT: 10 seconds (in search methods, for API calls)
- DOWNLOAD_TIMEOUT: 30 seconds (in download_pdf method)
- MIN_FILE_SIZE: 1000 bytes (in download_pdf method, minimum valid file size)
- SIMILARITY_THRESHOLD: 0.8 (in search_by_title method, for title matching)
- MAX_FILENAME_LENGTH: 200 characters (in create_filename method)

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

class OpenAlexFetcher:
    """
    Main class for fetching PDFs from OpenAlex API.

    This class handles the complete pipeline of:
    1. Loading publication metadata from a file
    2. Searching OpenAlex for each publication
    3. Finding available PDF download links
    4. Downloading PDFs with validation
    5. Tracking success/failure for each publication
    """

    def __init__(self):
        """
        Initialize the OpenAlex fetcher with configuration from environment variables.

        Sets up API credentials, creates necessary directories, and prepares
        HTTP headers for API requests. Like setting up a research library card
        and preparing your backpack before going to find books.
        """
        # Get file paths from environment variables - like reading your shopping list
        # The shopping list tells us which research papers we want to find and download
        self.input_file = os.getenv("PUBLICATIONS_FILE", "/app/data/publications.txt")
        self.output_dir = Path("/app/output/openalex")
        
        # Get our credentials for accessing the OpenAlex database
        # Think of this like having a library card that gives you special privileges
        self.openalex_token = os.getenv("OPENALEX_TOKEN", "")
        self.email = os.getenv("EMAIL", "researcher@example.com")

        # Create the folder where we'll save downloaded PDFs
        # Like making sure your downloads folder exists before you start shopping online
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Set up HTTP headers for API requests
        # This is like introducing yourself politely when you ask the librarian for help
        # We tell them who we are and give our contact info so they can help us better
        self.headers = {
            'User-Agent': f'PDFDownloader/1.0 (mailto:{self.email})',
        }
        if self.openalex_token:
            self.headers['Authorization'] = f'Bearer {self.openalex_token}'
            logger.info("Using OpenAlex API token for higher rate limits")

        # Track how many publications we process, find, download, etc.
        # Like keeping a scorecard during a game to see how well we're doing
        self.stats = {'processed': 0, 'found': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

    def init_tracking_file(self, publications):
        """
        Create a tracking file to record what happens to each publication.

        This file keeps a record of which publications were successfully downloaded,
        which failed, and why. It's like a checklist where you mark off items
        as you complete them, but also note any problems you encounter.

        Args:
            publications: List of publication dictionaries (not used in this method,
                         but passed for consistency with other methods)
        """
        # Create path for tracking file in publications subfolder
        self.tracking_file = Path(self.output_dir) / "publications" / "publications_openalex.txt"
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy the header line from input file (column names)
        # This ensures our tracking file has the same format as the input
        with open(self.input_file, 'r', encoding='utf-8') as f:
            header = f.readline()

        # Start the tracking file with just the header
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
        """Normalize text for consistent naming"""
        if not text:
            return ""
        # Remove special characters, convert to lowercase, remove extra spaces
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'\s+', '_', text.strip())
        return text

    def create_filename(self, pub_data, openalex_id):
        """
        Create a consistent, readable filename for the PDF.

        Think of this like creating a smart filing system for your research papers.
        Instead of naming files "paper1.pdf" or "download.pdf", we create descriptive names
        that tell you exactly what the paper is about, like:
        "10_1371_journal_pone_0123456_introduction_to_machine_learning_2023.pdf"

        The filename includes:
        - The DOI (like a paper's unique ID number)
        - Key words from the title (so you know what it's about)
        - Publication year (so you know how recent it is)

        Args:
            pub_data: Dictionary with publication metadata (like a paper's info card)
            openalex_id: OpenAlex identifier (currently not used in filename)

        Returns:
            String filename ending in .pdf that's easy to understand and organize
        """
        # Clean up the DOI for use in filename
        # DOIs are like ISBN numbers for books - they uniquely identify each paper
        # But they come with messy URL parts that we need to remove for filenames
        doi = pub_data.get('doi', '').lower()
        doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        doi = doi.replace('doi:', '').strip('/')
        doi = self.normalize_text(doi) if doi else 'no_doi'

        # Take first 5 words of title to keep filename reasonable length
        # This is like using a book title's first few words as a nickname
        # Instead of "A Comprehensive Study of Machine Learning Applications..."
        # we get "comprehensive_study_machine_learning_applications"
        title = pub_data.get('title', '')
        title_words = self.normalize_text(title).split('_')[:5]
        title_part = '_'.join(title_words) if title_words else 'no_title'

        # Get publication year - helps organize papers chronologically
        year = pub_data.get('year_pub', '').strip() or 'no_year'

        # Put it all together like building a descriptive label
        filename = f"{doi}_{title_part}_{year}.pdf"

        # Make sure filename isn't too long for the filesystem
        # Most filesystems have a 255 character limit, so we stay well under that
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

    def search_openalex(self, pub_data):
        """Search OpenAlex for publication"""
        # Try DOI first
        doi = pub_data.get('doi', '').strip()
        if doi:
            work = self.search_by_doi(doi)
            if work:
                return work
        
        # Try title if no DOI or DOI failed
        title = pub_data.get('title', '').strip()
        year = pub_data.get('year_pub', '').strip()
        if title:
            return self.search_by_title(title, year)
        
        return None

    def search_by_doi(self, doi):
        """Search OpenAlex by DOI"""
        try:
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            clean_doi = clean_doi.replace('doi:', '').strip('/')
            
            url = "https://api.openalex.org/works"
            params = {'filter': f'doi:{clean_doi}', 'per-page': 1}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results'):
                return data['results'][0]
            return None
        except Exception as e:
            logger.warning(f"DOI search failed: {e}")
            return None

    def search_by_title(self, title, year=None):
        """Search OpenAlex by title and verify year"""
        try:
            url = "https://api.openalex.org/works"
            params = {'search': title, 'per-page': 5}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            for work in data.get('results', []):
                work_title = work.get('title', '').lower()
                pub_year = str(work.get('publication_year', ''))
                
                # Check title similarity (simple word matching)
                title_similarity = self.calculate_similarity(title.lower(), work_title)
                year_match = not year or pub_year == year
                
                if title_similarity > 0.8 and year_match:
                    return work
            
            return None
        except Exception as e:
            logger.warning(f"Title search failed: {e}")
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
        Find all possible PDF download links from OpenAlex work data.

        Think of this like being a detective looking for a specific book.
        OpenAlex knows about many different places where the same paper might be available:
        - The publisher's official website (most reliable, but often behind paywall)
        - University repositories (usually free, good quality)
        - Preprint servers like arXiv (free, but might be earlier versions)
        - ResearchGate (free, but quality varies)

        We collect ALL possible sources and rank them by how trustworthy they are,
        like making a list of bookstores from "most likely to have what I want"
        to "worth a try but probably won't work".

        Args:
            work: OpenAlex work dictionary containing all the location data

        Returns:
            List of dictionaries with 'url', 'source', and 'priority' keys,
            sorted by priority (highest first, so we try the best sources first)
        """
        pdf_urls = []
        seen_urls = set()  # Keep track of URLs we've already found

        # Step 1: Check OpenAlex's "best open access location" first
        # This is like asking the librarian for their top recommendation
        best_oa = work.get('best_oa_location')
        if best_oa:
            # Try the direct PDF link first (highest priority)
            if best_oa.get('pdf_url'):
                url = best_oa['pdf_url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    pdf_urls.append({
                        'url': url,
                        'source': 'openalex_best_oa',
                        'priority': 100  # Highest priority
                    })
            # Also save the landing page URL (might redirect to PDF)
            if best_oa.get('url'):
                url = best_oa['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    pdf_urls.append({
                        'url': url,
                        'source': 'openalex_best_oa_landing',
                        'priority': 98  # Slightly lower priority
                    })
        
        # Step 2: Check general open access information
        # This is broader than the "best" location
        if work.get('open_access'):
            oa_data = work['open_access']

            # Try the open access URL
            if oa_data.get('oa_url'):
                url = oa_data['oa_url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Give higher priority if URL ends with .pdf
                    priority = 95 if url.lower().endswith('.pdf') else 93
                    pdf_urls.append({
                        'url': url,
                        'source': 'openalex_oa',
                        'priority': priority
                    })

            # Note: any_repository_has_fulltext indicates repositories have the paper,
            # but we'll find those specific links in the locations section below

        # Step 3: Check the primary location (usually the publisher's site)
        primary_loc = work.get('primary_location')
        if primary_loc:
            # Try direct PDF link from primary location
            if primary_loc.get('pdf_url'):
                url = primary_loc['pdf_url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    source_data = primary_loc.get('source', {})
                    source_name = source_data.get('display_name', 'unknown') if source_data else 'unknown'
                    pdf_urls.append({
                        'url': url,
                        'source': f'primary_{source_name}',
                        'priority': 92
                    })
            # Try landing page from primary location
            if primary_loc.get('url'):
                url = primary_loc['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    source_data = primary_loc.get('source', {})
                    source_name = source_data.get('display_name', 'unknown') if source_data else 'unknown'
                    pdf_urls.append({
                        'url': url,
                        'source': f'primary_landing_{source_name}',
                        'priority': 88
                    })
        
        # Step 4: Check all locations (repositories, archives, etc.)
        # This is the most comprehensive search - like checking every library in town
        for location in work.get('locations', []):
            source_data = location.get('source', {})
            source_name = source_data.get('display_name', 'unknown') if source_data else 'unknown'
            is_oa = location.get('is_oa', False)  # Is this location open access?

            # Get reliability score for this source (high for PMC, low for ResearchGate)
            base_priority = self.get_source_priority(source_name)

            # Try direct PDF link (best option for each source)
            if location.get('pdf_url'):
                url = location['pdf_url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Give bonus points if it's open access
                    priority = base_priority + (5 if is_oa else 0)
                    pdf_urls.append({
                        'url': url,
                        'source': source_name,
                        'priority': priority
                    })

            # Try special PDF URL field (some sources have this)
            if location.get('url_for_pdf'):
                url = location['url_for_pdf']
                if url not in seen_urls:
                    seen_urls.add(url)
                    priority = base_priority + (3 if is_oa else 0)
                    pdf_urls.append({
                        'url': url,
                        'source': f'{source_name}_pdf',
                        'priority': priority
                    })

            # Try landing page (might have PDF download link)
            if location.get('url'):
                url = location['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Landing pages get lower priority since they might not directly link to PDF
                    priority = base_priority - 10 + (3 if is_oa else 0)
                    pdf_urls.append({
                        'url': url,
                        'source': f'{source_name}_landing',
                        'priority': priority
                    })

        # Step 5: Last resort - try the DOI URL (might redirect to publisher page with PDF)
        if work.get('doi'):
            doi_url = work['doi']
            if doi_url not in seen_urls:
                seen_urls.add(doi_url)
                pdf_urls.append({
                    'url': doi_url,
                    'source': 'doi_redirect',
                    'priority': 50  # Very low priority
                })

        # Sort all found URLs by priority (best sources first)
        pdf_urls.sort(key=lambda x: x['priority'], reverse=True)

        # Tell the user what we found
        if pdf_urls:
            logger.info(f"  Found {len(pdf_urls)} potential PDF URLs")
            for i, pdf_info in enumerate(pdf_urls[:3]):  # Show top 3
                logger.debug(f"    {i+1}. {pdf_info['source']} (priority: {pdf_info['priority']})")

        return pdf_urls

    def get_source_priority(self, source):
        """
        Assign priority scores to different academic sources.
        
        Think of this like rating different stores based on how likely they are
        to actually have what you want when you get there. Some stores are
        reliable (always have good stuff, easy to access), while others are
        hit-or-miss (might have it, but often want you to sign up first).
        
        Priority scale: 95 = "Almost always works" down to 70 = "Worth a try"
        """
        if not source:
            return 70
        
        source_lower = source.lower()
        
        # Gold standard sources - like going to the official store
        # These almost always work and provide high-quality PDFs
        if 'pmc' in source_lower or 'pubmed' in source_lower:
            return 95  # PubMed Central - government-run, very reliable
        elif 'arxiv' in source_lower:
            return 94  # arXiv - preprint server, super reliable for STEM papers
        elif 'biorxiv' in source_lower or 'medrxiv' in source_lower:
            return 93  # bioRxiv/medRxiv - biology/medicine preprints, very good
        elif 'plos' in source_lower:
            return 92  # PLOS - open access publisher, always free
        elif 'frontiers' in source_lower:
            return 91  # Frontiers - another reliable open access publisher
            
        # Good commercial sources with open access versions
        elif 'nature' in source_lower and 'open' in source_lower:
            return 90  # Nature Open Access - high quality when available
        elif 'springer' in source_lower and 'open' in source_lower:
            return 89  # Springer Open - good quality, legitimate
            
        # Decent open access publishers
        elif 'mdpi' in source_lower:
            return 88  # MDPI - legitimate publisher, usually works
        elif 'hindawi' in source_lower:
            return 87  # Hindawi - another legitimate open access publisher
            
        # University and institutional repositories
        # Like borrowing from a friend's university library
        elif 'repository' in source_lower or 'archive' in source_lower:
            return 85  # Institutional repositories - usually good
        elif 'university' in source_lower or '.edu' in source_lower:
            return 84  # University websites - generally trustworthy
            
        # Data repositories - good for supplementary materials
        elif 'zenodo' in source_lower:
            return 83  # Zenodo - CERN-backed repository, very reliable
        elif 'figshare' in source_lower:
            return 82  # Figshare - academic data sharing, usually works
            
        # Social academic networks - hit or miss
        # Like asking a colleague if they have a copy
        elif 'researchgate' in source_lower:
            return 75  # ResearchGate - often requires login, but sometimes works
        elif 'academia.edu' in source_lower:
            return 74  # Academia.edu - similar issues to ResearchGate
        else:
            return 70  # Unknown source - worth trying but don't expect much

    def download_pdf(self, pdf_info, filename):
        """Download PDF from URL with better content type detection"""
        url = pdf_info['url']
        filepath = os.path.join(self.output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            logger.info(f"  File already exists: {filename}")
            self.stats['skipped'] += 1
            return True
        
        try:
            # Prepare headers
            headers = self.headers.copy()
            headers.update({
                'Accept': 'application/pdf, application/octet-stream, */*;q=0.8',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # First, try HEAD request to check content type
            try:
                head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                content_type = head_response.headers.get('content-type', '').lower()
                content_length = head_response.headers.get('content-length', '0')
                
                # Check if it's definitely HTML (skip download)
                if 'text/html' in content_type and 'pdf' not in url.lower():
                    logger.debug(f"  Skipping HTML page from {pdf_info['source']}")
                    return False
                
                # Check if file is too small (probably an error page)
                if content_length and int(content_length) < 1000:
                    logger.debug(f"  File too small ({content_length} bytes) from {pdf_info['source']}")
                    return False
            except:
                # If HEAD fails, continue with GET
                pass
            
            # Download with streaming
            response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type from GET response
            content_type = response.headers.get('content-type', '').lower()
            
            # List of acceptable content types for PDFs
            pdf_content_types = [
                'application/pdf',
                'application/octet-stream',
                'application/x-pdf',
                'application/download',
                'binary/octet-stream',
                'application/force-download'
            ]
            
            # Check if content might be PDF
            is_likely_pdf = any(ct in content_type for ct in pdf_content_types)
            
            # If it's HTML, skip it
            if 'text/html' in content_type and not is_likely_pdf:
                logger.warning(f"  Got HTML instead of PDF from {pdf_info['source']}")
                return False
            
            # Download to temporary file first
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Check file size
            file_size = os.path.getsize(temp_filepath)
            if file_size < 1000:  # Less than 1KB is suspicious
                os.remove(temp_filepath)
                logger.warning(f"  Downloaded file too small ({file_size} bytes) from {pdf_info['source']}")
                return False
            
            # Check if it's actually a PDF by reading first bytes
            with open(temp_filepath, 'rb') as f:
                header = f.read(min(1024, file_size))
            
            # Check for PDF signature or common PDF patterns
            if header.startswith(b'%PDF'):
                # It's a PDF!
                os.rename(temp_filepath, filepath)
                logger.info(f"  Downloaded: {filename} ({file_size:,} bytes) from {pdf_info['source']}")
                self.stats['downloaded'] += 1
                return True
            elif b'<!DOCTYPE html' in header or b'<html' in header:
                # It's HTML
                os.remove(temp_filepath)
                logger.warning(f"  Got HTML content from {pdf_info['source']}")
                return False
            elif content_type and 'pdf' in content_type:
                # Content type says PDF but doesn't start with %PDF - might still be valid
                os.rename(temp_filepath, filepath)
                logger.info(f"  Downloaded: {filename} ({file_size:,} bytes) from {pdf_info['source']} [non-standard PDF]")
                self.stats['downloaded'] += 1
                return True
            else:
                # Unknown content
                os.remove(temp_filepath)
                logger.warning(f"  Unknown content type from {pdf_info['source']}")
                return False
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"  404 Not Found from {pdf_info['source']}")
            else:
                logger.warning(f"  HTTP error {e.response.status_code} from {pdf_info['source']}")
            return False
        except Exception as e:
            logger.warning(f"  Download failed from {pdf_info['source']}: {str(e)[:100]}")
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return False

    def process_publications(self):
        """Main processing function"""
        logger.info("Starting OpenAlex PDF download")
        
        publications = self.load_publications()
        if not publications:
            return
        
        # Initialize tracking file
        self.init_tracking_file(publications)
        
        for i, pub_data in enumerate(publications, 1):
            pub_id = pub_data.get('pub_id', f'pub_{i}')
            logger.info(f"Processing {i}/{len(publications)}: {pub_id}")
            
            self.stats['processed'] += 1
            
            # Search OpenAlex
            work = self.search_openalex(pub_data)
            if not work:
                logger.info(f"  Not found in OpenAlex")
                self.update_tracking_file(pub_data, 'failed', 'Not found in OpenAlex')
                continue
            
            self.stats['found'] += 1
            openalex_id = work.get('id', '').replace('https://openalex.org/', '')
            
            # Get PDF URLs
            pdf_urls = self.get_pdf_urls(work)
            if not pdf_urls:
                logger.info(f"  No PDF URLs found")
                self.update_tracking_file(pub_data, 'failed', 'No PDF URLs available')
                continue
            
            # Create filename
            filename = self.create_filename(pub_data, openalex_id)
            
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
            
            # Rate limiting
            time.sleep(0.1)
            
            # Progress update
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(publications)} - "
                        f"Found: {self.stats['found']}, Downloaded: {self.stats['downloaded']}")

    def print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*50)
        logger.info("OPENALEX FETCHER SUMMARY")
        logger.info("="*50)
        for key, value in self.stats.items():
            logger.info(f"{key.capitalize()}: {value:,}")
        
        if self.stats['found'] > 0:
            success_rate = (self.stats['downloaded'] / self.stats['found']) * 100
            logger.info(f"Download Success Rate: {success_rate:.1f}%")

def main():
    fetcher = OpenAlexFetcher()
    fetcher.process_publications()
    fetcher.print_summary()

if __name__ == "__main__":
    main()

print("✅ OpenAlex PDF fetcher module loaded successfully")
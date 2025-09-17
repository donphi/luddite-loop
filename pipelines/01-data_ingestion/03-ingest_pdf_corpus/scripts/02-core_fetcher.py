#!/usr/bin/env python3
"""
Stage 2: CORE PDF Fetcher
Simple script to download PDFs from CORE API with retry logic
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
    def __init__(self):
        self.input_file = os.getenv("PUBLICATIONS_FILE", "/app/data/publications.txt")
        self.output_dir = Path("/app/output/core")
        self.core_api_key = os.getenv('CORE_API_KEY', '')
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # API setup
        self.base_url = "https://api.core.ac.uk/v3"
        self.headers = {
            'Authorization': f'Bearer {self.core_api_key}',
            'Content-Type': 'application/json'
        }
        
        if not self.core_api_key:
            logger.warning("No CORE API key provided - this may limit results")
        else:
            logger.info("Using CORE API key")
        
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
        """Make API request with exponential backoff retry for rate limits and server errors"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()
                return response
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = min(60, (2 ** attempt) * 5)  # 5, 10, 20, max 60 seconds
                    logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # <-- ADD THIS
                    # Server errors - retry with backoff
                    wait_time = min(30, (2 ** attempt) * 2)  # 2, 4, 8 seconds
                    logger.warning(f"Server error ({e.response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    # Other HTTP errors (4xx) - don't retry
                    raise
            except Exception as e:
                # Network errors, timeouts etc - could retry but let's just fail
                logger.warning(f"Request failed: {e}")
                return None
        
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
        """Extract PDF URLs from CORE work"""
        pdf_urls = []
        
        # Check for fullText download URL
        if work.get('fullTextIdentifier'):
            pdf_urls.append({
                'url': work['fullTextIdentifier'],
                'source': 'core_fulltext',
                'priority': 90
            })
        
        # Check for download URL
        if work.get('downloadUrl'):
            pdf_urls.append({
                'url': work['downloadUrl'],
                'source': 'core_download',
                'priority': 85
            })
        
        # Check links for PDFs
        for link in work.get('links', []):
            if link.get('url', '').endswith('.pdf'):
                pdf_urls.append({
                    'url': link['url'],
                    'source': 'core_link',
                    'priority': 80
                })
        
        # Sort by priority
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
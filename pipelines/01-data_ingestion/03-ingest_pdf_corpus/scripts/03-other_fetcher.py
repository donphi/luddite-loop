# ============================================================================
# FILE: scripts/03-other_fetcher.py
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/scripts/03-other_fetcher.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Fetches PDF documents from additional open access sources (Unpaywall, PMC, arXiv)
# ============================================================================

"""
MODULE OVERVIEW:
This module downloads PDF documents from various open access sources beyond OpenAlex and CORE.
It searches Unpaywall (for DOI-based OA discovery), PubMed Central (PMC), and arXiv
to find additional PDF sources for publications.

CLASSES:
- OtherSourcesFetcher: Main class for handling multiple OA source interactions and PDF downloads

METHODS:
- __init__(): Initializes the fetcher with environment variables and creates output directories
- init_tracking_file(): Creates a tracking file to record download status for each publication
- update_tracking_file(): Updates the tracking file with success/failure status
- normalize_text(): Cleans text for consistent filename generation
- create_filename(): Generates standardized PDF filenames from publication metadata
- load_publications(): Reads publication data from input TSV file
- search_unpaywall(): Searches Unpaywall API for open access PDFs by DOI
- search_pmc(): Searches PubMed Central for open access PDFs
- search_pmc_by_doi(): Searches PMC specifically by DOI identifier
- search_pmc_by_title(): Searches PMC by title with year verification
- search_arxiv(): Searches arXiv preprint repository for papers
- calculate_similarity(): Computes text similarity for title matching
- search_all_sources(): Orchestrates searches across all supported sources
- download_pdf(): Downloads PDF from URL with content validation
- process_publications(): Main orchestration method that processes all publications
- print_summary(): Displays final statistics of the download operation

ROUTES:
- N/A (This is a data processing module, not a web service)

HYPERPARAMETERS:
- REQUEST_TIMEOUT: 10 seconds (in search methods, for API calls)
- DOWNLOAD_TIMEOUT: 30 seconds (in download_pdf method)
- MIN_FILE_SIZE: 1000 bytes (in download_pdf method, minimum valid file size)
- TITLE_SIMILARITY_THRESHOLD: 0.7 (in search_pmc_by_title method, for PMC title matching)
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

class OtherSourcesFetcher:
    def __init__(self):
        self.input_file = os.getenv("PUBLICATIONS_FILE", "/app/data/publications.txt")
        self.output_dir = Path("/app/output/other")
        self.email = os.getenv('EMAIL', 'researcher@example.com')
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # API setup
        self.headers = {
            'User-Agent': f'PDFDownloader/1.0 (mailto:{self.email})',
        }
        
        self.stats = {'processed': 0, 'found': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

    def init_tracking_file(self, publications):
        """Initialize the tracking file with header from original file"""
        self.tracking_file = Path(self.output_dir) / "publications" / "publications_other.txt"
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
        """Normalize text for consistent naming (same as other stages)"""
        if not text:
            return ""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'\s+', '_', text.strip())
        return text

    def create_filename(self, pub_data, source_id):
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

    def search_unpaywall(self, pub_data):
        """Search Unpaywall for open access PDFs"""
        doi = pub_data.get('doi', '').strip()
        if not doi:
            return None
        
        try:
            # Clean DOI
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            clean_doi = clean_doi.replace('doi:', '').strip('/')
            
            url = f"https://api.unpaywall.org/v2/{clean_doi}?email={self.email}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if there are open access locations
            if data.get('is_oa') and data.get('oa_locations'):
                for location in data['oa_locations']:
                    if location.get('url_for_pdf'):
                        return {
                            'url': location['url_for_pdf'],
                            'source': f"unpaywall_{location.get('host_type', 'unknown')}",
                            'priority': 85
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Unpaywall search failed: {e}")
            return None

    def search_pmc(self, pub_data):
        """Search PMC for open access PDFs"""
        # Try by DOI first
        doi = pub_data.get('doi', '').strip()
        if doi:
            result = self.search_pmc_by_doi(doi)
            if result:
                return result
        
        # Try by title if no DOI success
        title = pub_data.get('title', '').strip()
        year = pub_data.get('year_pub', '').strip()
        if title:
            return self.search_pmc_by_title(title, year)
        
        return None

    def search_pmc_by_doi(self, doi):
        """Search PMC by DOI"""
        try:
            clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            clean_doi = clean_doi.replace('doi:', '').strip('/')
            
            # Search PMC
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pmc',
                'term': f'"{clean_doi}"[doi]',
                'retmode': 'json',
                'retmax': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if id_list:
                pmc_id = id_list[0]
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                return {
                    'url': pdf_url,
                    'source': 'pmc',
                    'priority': 95
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"PMC DOI search failed: {e}")
            return None

    def search_pmc_by_title(self, title, year=None):
        """Search PMC by title"""
        try:
            search_term = f'"{title}"'
            if year:
                search_term += f' AND {year}[pdat]'
            
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pmc',
                'term': search_term,
                'retmode': 'json',
                'retmax': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if id_list:
                # Get details for first result to verify title match
                pmc_id = id_list[0]
                
                # Get summary to check title
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                summary_params = {
                    'db': 'pmc',
                    'id': pmc_id,
                    'retmode': 'json'
                }
                
                summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                summary_data = summary_response.json()
                
                # Simple title check
                result_title = summary_data.get('result', {}).get(pmc_id, {}).get('title', '')
                title_similarity = self.calculate_similarity(title.lower(), result_title.lower())
                
                if title_similarity > 0.7:  # More lenient for PMC
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                    return {
                        'url': pdf_url,
                        'source': 'pmc',
                        'priority': 95
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"PMC title search failed: {e}")
            return None

    def search_arxiv(self, pub_data):
        """Search arXiv for papers"""
        title = pub_data.get('title', '').strip()
        if not title:
            return None
        
        try:
            # arXiv search
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'ti:"{title}"',
                'max_results': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            content = response.text
            if '<entry>' in content:
                # Extract first PDF link
                import re
                pdf_match = re.search(r'<link title="pdf" href="([^"]+)"', content)
                if pdf_match:
                    pdf_url = pdf_match.group(1)
                    return {
                        'url': pdf_url,
                        'source': 'arxiv',
                        'priority': 90
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
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

    def search_all_sources(self, pub_data):
        """Search all other sources for PDFs"""
        sources = []
        
        # Try Unpaywall (best for DOI-based search)
        unpaywall_result = self.search_unpaywall(pub_data)
        if unpaywall_result:
            sources.append(unpaywall_result)
        
        # Try PMC
        pmc_result = self.search_pmc(pub_data)
        if pmc_result:
            sources.append(pmc_result)
        
        # Try arXiv for preprints
        arxiv_result = self.search_arxiv(pub_data)
        if arxiv_result:
            sources.append(arxiv_result)
        
        # Sort by priority
        sources.sort(key=lambda x: x['priority'], reverse=True)
        return sources

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
            headers = self.headers.copy()
            headers.update({
                'Accept': 'application/pdf,*/*',
                'User-Agent': 'Mozilla/5.0 (compatible; PDFDownloader/1.0)'
            })
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                logger.warning(f"  Got HTML instead of PDF from {pdf_info['source']}")
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
        logger.info("Starting Other Sources PDF download")
        
        publications = self.load_publications()
        if not publications:
            return
        
        # Initialize tracking file
        self.init_tracking_file(publications)
        
        for i, pub_data in enumerate(publications, 1):
            pub_id = pub_data.get('pub_id', f'pub_{i}')
            logger.info(f"Processing {i}/{len(publications)}: {pub_id}")
            
            self.stats['processed'] += 1
            
            # Search all other sources
            sources = self.search_all_sources(pub_data)
            if not sources:
                logger.info(f"  Not found in any other sources")
                self.update_tracking_file(pub_data, 'failed', 'Not found in any source')
                continue
            
            self.stats['found'] += 1
            
            # Create filename (same format as other stages)
            filename = self.create_filename(pub_data, 'other')
            
            # Check if already exists
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"  File already exists: {filename}")
                self.stats['skipped'] += 1
                self.update_tracking_file(pub_data, 'success')
                continue
            
            # Try to download from sources (in priority order)
            downloaded = False
            last_error = "Download failed from all sources"
            for pdf_info in sources:
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
            time.sleep(0.5)  # Slower for these APIs
            
            # Progress update
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(publications)} - "
                        f"Found: {self.stats['found']}, Downloaded: {self.stats['downloaded']}")

    def print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*50)
        logger.info("OTHER SOURCES FETCHER SUMMARY")
        logger.info("="*50)
        for key, value in self.stats.items():
            logger.info(f"{key.capitalize()}: {value:,}")
        
        if self.stats['found'] > 0:
            success_rate = (self.stats['downloaded'] / self.stats['found']) * 100
            logger.info(f"Download Success Rate: {success_rate:.1f}%")

def main():
    fetcher = OtherSourcesFetcher()
    fetcher.process_publications()
    fetcher.print_summary()

if __name__ == "__main__":
    main()

print("✅ Other sources PDF fetcher module loaded successfully")
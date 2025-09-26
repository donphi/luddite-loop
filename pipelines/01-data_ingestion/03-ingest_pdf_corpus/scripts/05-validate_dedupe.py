#!/usr/bin/env python3
# ============================================================================
# FILE: scripts/04-validate_dedupe.py
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/scripts/04-validate_dedupe.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Validates downloaded PDFs and removes duplicates using multiple methods
# ============================================================================

"""
MODULE OVERVIEW:
This module validates all PDFs downloaded from various sources (including manual downloads)
and removes duplicates using multiple deduplication strategies including content hashing,
metadata extraction, and text similarity comparison.

CLASSES:
- PDFValidator: Main class for PDF validation and deduplication

METHODS:
- __init__(): Initializes the validator with directory paths and loads publication data
- load_publications(): Reads publication metadata for DOI mapping
- load_manual_metadata(): Loads metadata from manual_download.txt for manual PDFs
- create_doi_mapping(): Creates mapping from DOI to publication ID
- normalize_doi(): Standardizes DOI format for consistent comparison
- validate_pdf(): Checks if a PDF is valid
- extract_pdf_metadata(): Extracts metadata directly from PDF file
- extract_text_from_pdf(): Extracts text content for similarity comparison
- calculate_file_hash(): Computes SHA256 hash of PDF content
- calculate_text_similarity(): Computes Jaccard similarity between text samples
- extract_metadata_from_filename(): Parses DOI and metadata from filenames
- identify_pdf(): Attempts to identify a PDF using multiple methods
- group_pdfs_by_identity(): Groups PDFs using multiple deduplication strategies
- select_best_pdf(): Chooses the highest quality PDF from duplicates
- process_all_pdfs(): Main orchestration method
- generate_report(): Creates detailed validation report
- print_summary(): Displays final statistics

HYPERPARAMETERS:
- MIN_FILE_SIZE: 1000 bytes (minimum valid file size)
- MIN_PAGE_COUNT: 2 (minimum pages for valid PDF)
- MIN_TEXT_LENGTH: 10 characters (minimum extractable text)
- TEXT_SIMILARITY_THRESHOLD: 0.85 (for text-based deduplication)
- CONTENT_SAMPLE_SIZE: 5000 characters (for text comparison)

DEPENDENCIES:
- PyPDF2: For PDF reading and validation
- hashlib: For content-based hashing
- pathlib: For file operations
- csv: For reading metadata
- logging: For operation tracking
- shutil: For moving files
- collections.defaultdict: For grouping
"""

import os
import csv
import shutil
import logging
import hashlib
import re
import PyPDF2
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFValidator:
    """
    Validates and deduplicates all downloaded PDFs using multiple methods.
    
    Think of this like having a quality control inspector at a book warehouse.
    After collecting books from many different suppliers (OpenAlex, CORE, etc.),
    we need to:
    
    1. Check each book to make sure it's not corrupted (validation)
    2. Remove duplicates when we accidentally got the same book from multiple suppliers
    3. Pick the best copy when we have duplicates
    4. Organize everything into "good" and "problematic" piles
    
    We use multiple ways to detect duplicates:
    - Exact file matching (identical files)
    - DOI matching (same research paper ID)
    - Text similarity (same content, different files)
    """
    
    def __init__(self):
        """
        Set up the PDF validation and deduplication system.
        
        This is like organizing a quality control station with:
        - Input bins (where PDFs come from different sources)
        - Sorting areas (valid, invalid, duplicates)
        - Reference materials (publication metadata)
        - Quality tracking (statistics)
        """
        # Where to find our publication metadata (the master list)
        self.input_file = "/app/data/publications.txt"
        
        # All the places where we might have downloaded PDFs
        # Like checking different warehouses for inventory
        self.source_dirs = [
            "/app/output/openalex",      # From OpenAlex database
            "/app/output/core",          # From CORE database
            "/app/output/other",         # From other sources (Unpaywall, PMC, arXiv)
            "/app/output/manual",        # Manually downloaded files
            "/app/output/manual_renamed" # Manually downloaded and properly named
        ]
        
        # Where to sort the PDFs after processing
        # Like having labeled bins for different types of books
        self.valid_dir = "/app/output/valid_pdfs"      # Good PDFs, ready to use
        self.invalid_dir = "/app/output/invalid_pdfs"   # Corrupted or problematic PDFs
        self.duplicates_dir = "/app/output/duplicate_pdfs"  # Extra copies we don't need
        self.output_dir = "/app/output"
        
        # Create the sorting bins if they don't exist
        for dir_path in [self.valid_dir, self.invalid_dir, self.duplicates_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Load our reference materials (publication metadata)
        self.publications = self.load_publications()
        self.doi_to_pubid = self.create_doi_mapping()
        
        # Load information about manually downloaded files
        self.manual_metadata = self.load_manual_metadata()
        
        # Set up our duplicate detection systems
        # Like having different methods to check if books are the same
        self.pdf_fingerprints = {}  # File content signatures
        self.text_samples = {}      # Text excerpts for comparison
        
        # Keep track of our quality control statistics
        # Like a dashboard showing how well the sorting process is going
        self.stats = {
            'total_pdfs': 0,
            'valid_pdfs': 0,
            'invalid_pdfs': 0,
            'duplicates_removed': 0,
            'final_unique_pdfs': 0,
            'hash_duplicates': 0,    # Exact file duplicates
            'doi_duplicates': 0,     # Same DOI duplicates
            'text_duplicates': 0     # Similar content duplicates
        }

    def load_publications(self):
        """Load publication data"""
        publications = {}
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    pub_id = row.get('pub_id', '').strip()
                    if pub_id:
                        publications[pub_id] = row
            logger.info(f"Loaded {len(publications)} publications")
            return publications
        except Exception as e:
            logger.error(f"Error loading publications: {e}")
            return {}

    def load_manual_metadata(self):
        """Load metadata for manually downloaded PDFs"""
        manual_metadata = {}
        manual_file = Path("/app/output/manual_download.txt")
        
        if not manual_file.exists():
            logger.info("No manual download metadata file found")
            return manual_metadata
        
        try:
            with open(manual_file, 'r', encoding='utf-8') as f:
                # Check if has header
                first_line = f.readline()
                if 'pub_id' in first_line.lower():
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        pub_id = row.get('pub_id', '')
                        if pub_id:
                            manual_metadata[pub_id] = {
                                'doi': row.get('doi', ''),
                                'title': row.get('title', ''),
                                'url': row.get('url', '')
                            }
            logger.info(f"Loaded metadata for {len(manual_metadata)} manual downloads")
        except Exception as e:
            logger.error(f"Error loading manual metadata: {e}")
        
        return manual_metadata

    def create_doi_mapping(self):
        """Create DOI to pub_id mapping for deduplication"""
        doi_mapping = {}
        
        # From main publications
        for pub_id, pub_data in self.publications.items():
            doi = pub_data.get('doi', '').strip()
            if doi and doi.lower() != 'nan':
                clean_doi = self.normalize_doi(doi)
                if clean_doi:
                    doi_mapping[clean_doi] = pub_id
        
        # From manual metadata
        for pub_id, metadata in self.manual_metadata.items():
            doi = metadata.get('doi', '').strip()
            if doi and doi.lower() != 'nan' and doi != 'no_doi':
                clean_doi = self.normalize_doi(doi)
                if clean_doi:
                    doi_mapping[clean_doi] = pub_id
        
        logger.info(f"Created DOI mapping for {len(doi_mapping)} publications")
        return doi_mapping

    def normalize_doi(self, doi):
        """Normalize DOI for consistent comparison"""
        if not doi:
            return ""
        
        doi = doi.lower().strip()
        doi = doi.replace('https://doi.org/', '')
        doi = doi.replace('http://dx.doi.org/', '')
        doi = doi.replace('doi:', '')
        doi = doi.strip('/')
        
        return doi

    def validate_pdf(self, pdf_path):
        """
        Check if a PDF file is valid and extract useful information.
        
        This is like being a book inspector at a warehouse. For each book,
        we need to check:
        - Is it actually a complete book or just an empty cover?
        - Can we open it and read it?
        - Is it damaged or corrupted?
        - What's the book about? (extract title, DOI, sample text)
        - How can we identify it later? (create a unique fingerprint)
        
        We're pretty strict about what counts as "valid" because we want
        to make sure researchers get useful, complete papers.
        """
        # Start with a report card for this PDF
        result = {
            'filepath': str(pdf_path),
            'filename': pdf_path.name,
            'is_valid': False,
            'page_count': 0,
            'file_size_bytes': 0,
            'error': None,
            'hash': None,                # Unique fingerprint for duplicate detection
            'extracted_doi': None,       # Paper's unique ID if we can find it
            'extracted_title': None,     # Paper title if we can extract it
            'text_sample': None          # Sample text for similarity comparison
        }
        
        try:
            # Check file size first - is this actually a substantial file?
            # Like checking if a book feels suspiciously light
            file_size = pdf_path.stat().st_size
            result['file_size_bytes'] = file_size
            
            if file_size < 1000:  # Less than 1KB is definitely not a real paper
                result['error'] = 'File too small (< 1KB)'
                return result
            
            # Create a unique fingerprint for this file
            # Like taking a DNA sample to identify duplicates later
            result['hash'] = self.calculate_file_hash(pdf_path)
            
            # Try to open and read the PDF
            # Like trying to open a book and flip through the pages
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Check if the PDF is password protected
                # Like finding a locked diary - we can't use it
                if pdf_reader.is_encrypted:
                    result['error'] = 'PDF is encrypted'
                    return result
                
                # Count the pages
                # Research papers should have multiple pages
                page_count = len(pdf_reader.pages)
                result['page_count'] = page_count
                
                if page_count == 0:
                    result['error'] = 'PDF has 0 pages'
                    return result
                
                if page_count < 2:
                    result['error'] = f'PDF too short ({page_count} page)'
                    return result
                
                # Try to extract metadata (title, DOI, etc.)
                # Like reading the book's title page and copyright info
                metadata = self.extract_pdf_metadata(pdf_reader)
                result['extracted_doi'] = metadata.get('doi')
                result['extracted_title'] = metadata.get('title')
                
                # Extract a sample of text for similarity comparison
                # Like reading the first few paragraphs to understand what it's about
                text_sample = self.extract_text_from_pdf(pdf_reader, max_chars=5000)
                result['text_sample'] = text_sample
                
                # Make sure we can actually read meaningful text
                # Sometimes PDFs are just scanned images with no extractable text
                if not text_sample or len(text_sample.strip()) < 10:
                    result['error'] = 'Cannot extract meaningful text'
                    return result
                
                # If we made it this far, the PDF is good to use!
                result['is_valid'] = True
                
        except PyPDF2.errors.PdfReadError as e:
            result['error'] = f'PDF read error: {str(e)}'
        except Exception as e:
            result['error'] = f'Validation error: {str(e)}'
        
        return result

    def extract_pdf_metadata(self, pdf_reader):
        """Extract metadata directly from PDF"""
        metadata = {}
        
        try:
            # Try to get metadata from PDF
            pdf_metadata = pdf_reader.metadata if hasattr(pdf_reader, 'metadata') else {}
            
            # Look for DOI in metadata
            for key in ['/doi', '/DOI', '/Subject', '/Keywords']:
                if pdf_metadata and key in pdf_metadata:
                    value = str(pdf_metadata[key])
                    # Search for DOI pattern
                    doi_pattern = r'10\.\d{4,}(?:\.\d+)*\/[-._;()\/:a-zA-Z0-9]+'
                    match = re.search(doi_pattern, value)
                    if match:
                        metadata['doi'] = match.group()
                        break
            
            # Get title
            if pdf_metadata and '/Title' in pdf_metadata:
                metadata['title'] = str(pdf_metadata['/Title'])
            
            # Also try to extract DOI from first page text
            if not metadata.get('doi'):
                try:
                    first_page_text = pdf_reader.pages[0].extract_text()[:2000]
                    doi_pattern = r'10\.\d{4,}(?:\.\d+)*\/[-._;()\/:a-zA-Z0-9]+'
                    match = re.search(doi_pattern, first_page_text)
                    if match:
                        metadata['doi'] = match.group()
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        
        return metadata

    def extract_text_from_pdf(self, pdf_reader, max_chars=5000):
        """Extract text sample from PDF for similarity comparison"""
        text_sample = ""
        
        try:
            # Extract text from first few pages
            for i, page in enumerate(pdf_reader.pages[:3]):  # First 3 pages
                text = page.extract_text()
                text_sample += text + " "
                
                if len(text_sample) >= max_chars:
                    break
            
            # Clean and normalize text
            text_sample = ' '.join(text_sample.split())  # Remove extra whitespace
            text_sample = text_sample[:max_chars]
            
        except Exception as e:
            logger.debug(f"Error extracting text: {e}")
        
        return text_sample

    def calculate_file_hash(self, pdf_path):
        """Calculate SHA256 hash of PDF file for exact duplicate detection"""
        sha256_hash = hashlib.sha256()
        
        with open(pdf_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()

    def calculate_text_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two text samples"""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def extract_metadata_from_filename(self, filename):
        """Extract DOI and other metadata from filename"""
        # Expected formats:
        # Standard: {doi}_{title_words}_{year}.pdf
        # Manual: manual_{pub_id}_{title_words}.pdf
        
        name_without_ext = filename.replace('.pdf', '')
        parts = name_without_ext.split('_')
        
        if len(parts) < 2:
            return None
        
        # Check if it's a manual file
        if parts[0] == 'manual' and len(parts) > 1:
            # Try to get pub_id and match with manual metadata
            pub_id = '_'.join(parts[:2]) if len(parts) > 1 else parts[1]
            if pub_id in self.manual_metadata:
                return {
                    'pub_id': pub_id,
                    'doi': self.manual_metadata[pub_id].get('doi'),
                    'title': self.manual_metadata[pub_id].get('title')
                }
        
        # Standard format with DOI
        doi_part = parts[0]
        if doi_part != 'no_doi':
            return {
                'doi': doi_part,
                'filename_parts': parts
            }
        
        return None

    def identify_pdf(self, pdf_info):
        """Attempt to identify a PDF using multiple methods"""
        identity = {
            'method': None,
            'key': None,
            'confidence': 0.0
        }
        
        # Method 1: File hash (100% confidence - exact duplicate)
        if pdf_info['hash']:
            identity = {
                'method': 'hash',
                'key': f"hash_{pdf_info['hash']}",
                'confidence': 1.0
            }
            return identity
        
        # Method 2: Extracted DOI from PDF metadata (95% confidence)
        if pdf_info['extracted_doi']:
            normalized_doi = self.normalize_doi(pdf_info['extracted_doi'])
            if normalized_doi:
                identity = {
                    'method': 'extracted_doi',
                    'key': f"doi_{normalized_doi}",
                    'confidence': 0.95
                }
                return identity
        
        # Method 3: DOI from filename (90% confidence)
        metadata = self.extract_metadata_from_filename(pdf_info['filename'])
        if metadata and metadata.get('doi'):
            normalized_doi = self.normalize_doi(metadata['doi'])
            if normalized_doi:
                identity = {
                    'method': 'filename_doi',
                    'key': f"doi_{normalized_doi}",
                    'confidence': 0.90
                }
                return identity
        
        # Method 4: Manual pub_id mapping (85% confidence)
        if metadata and metadata.get('pub_id'):
            identity = {
                'method': 'manual_pubid',
                'key': f"pubid_{metadata['pub_id']}",
                'confidence': 0.85
            }
            return identity
        
        # Method 5: Text similarity (variable confidence based on similarity score)
        # This will be done in the grouping phase
        
        # Default: Use filename as key (low confidence)
        identity = {
            'method': 'filename',
            'key': f"filename_{pdf_info['filename']}",
            'confidence': 0.1
        }
        
        return identity

    def group_pdfs_by_identity(self, valid_pdfs):
        """
        Group PDFs that represent the same research paper.
        
        This is like organizing a pile of books where you accidentally got
        multiple copies of the same book from different stores. We need to:
        
        1. First pass: Group books that are obviously the same (same ISBN,
           same file, same DOI)
        2. Second pass: For remaining books, read a bit of each to see if
           the content is the same even if the covers look different
        
        We use different confidence levels:
        - 95%+: "These are definitely the same" (same DOI, same file)
        - 85%+: "These are probably the same" (similar filenames)
        - Text similarity: "Let me read both and compare"
        """
        groups = defaultdict(list)
        text_similarity_candidates = []
        
        # First pass: Group by exact identifiers
        # Like sorting books by ISBN number - if the ISBN matches, they're the same book
        for pdf_info in valid_pdfs:
            identity = self.identify_pdf(pdf_info)
            
            if identity['confidence'] >= 0.85:
                # High confidence match - we're pretty sure these are the same
                groups[identity['key']].append(pdf_info)
            else:
                # Low confidence - we need to check content similarity
                text_similarity_candidates.append(pdf_info)
        
        # Second pass: Text similarity for remaining PDFs
        # Like reading the first few paragraphs of each book to see if they're the same
        logger.info(f"Checking text similarity for {len(text_similarity_candidates)} PDFs...")
        
        for candidate in text_similarity_candidates:
            matched = False
            
            if candidate['text_sample']:
                # Compare this candidate's text with existing groups
                for group_key, group_pdfs in groups.items():
                    if group_pdfs and group_pdfs[0].get('text_sample'):
                        similarity = self.calculate_text_similarity(
                            candidate['text_sample'],
                            group_pdfs[0]['text_sample']
                        )
                        
                        if similarity >= 0.85:  # 85% text similarity = probably same paper
                            logger.info(f"  Text similarity match ({similarity:.2f}): "
                                      f"{candidate['filename']} -> {group_key}")
                            groups[group_key].append(candidate)
                            matched = True
                            self.stats['text_duplicates'] += 1
                            break
            
            if not matched:
                # This appears to be a unique document
                unique_key = f"unique_{candidate['filename']}"
                groups[unique_key].append(candidate)
        
        return groups

    def select_best_pdf(self, pdf_group):
        """
        Choose the best PDF when we have multiple copies of the same paper.
        
        This is like choosing the best copy when you have the same book
        from different stores. We prefer:
        - Longer books (more pages = more complete)
        - Bigger files (usually higher quality)
        - Books with clear identification (DOI extracted)
        - Books from reputable stores (OpenAlex > PMC > CORE)
        - Properly organized books (not manually downloaded with messy names)
        
        Think of it like a scoring system where we give points for quality
        indicators, then pick the highest-scoring copy.
        """
        if len(pdf_group) == 1:
            return pdf_group[0], []  # Only one copy, so it's automatically the best
        
        # Score each PDF based on quality indicators
        scored_pdfs = []
        for pdf in pdf_group:
            score = 0
            
            # Prefer more pages (more complete content)
            # Like preferring a complete book over a partial one
            score += pdf['page_count'] * 10
            
            # Prefer larger file size up to a reasonable limit
            # Bigger usually means higher quality, but not always
            score += min(pdf['file_size_bytes'] / 100000, 50)  # Cap bonus at 5MB
            
            # Prefer PDFs where we successfully extracted the DOI
            # This indicates good metadata and proper formatting
            if pdf.get('extracted_doi'):
                score += 20
            
            # Prefer non-manual sources because they have proper naming
            # Manual downloads often have generic names like "download.pdf"
            if not pdf['filename'].startswith('manual_'):
                score += 15
            
            # Prefer PDFs from more reliable sources
            # Like preferring books from well-known, reputable publishers
            if 'openalex' in pdf['filepath'].lower():
                score += 10  # OpenAlex is very comprehensive and reliable
            elif 'pmc' in pdf['filepath'].lower():
                score += 8   # PubMed Central is government-run, very trustworthy
            elif 'core' in pdf['filepath'].lower():
                score += 5   # CORE is good but sometimes has quality issues
            
            scored_pdfs.append((pdf, score))
        
        # Sort by score (highest first) and pick the winner
        scored_pdfs.sort(key=lambda x: x[1], reverse=True)
        
        best_pdf = scored_pdfs[0][0]
        duplicates = [pdf for pdf, _ in scored_pdfs[1:]]  # All the others are duplicates
        
        # Keep track of what type of duplicates we found for reporting
        if best_pdf.get('hash') == duplicates[0].get('hash') if duplicates else False:
            self.stats['hash_duplicates'] += len(duplicates)
        elif best_pdf.get('extracted_doi') or duplicates[0].get('extracted_doi') if duplicates else False:
            self.stats['doi_duplicates'] += len(duplicates)
        
        return best_pdf, duplicates

    def process_all_pdfs(self):
        """Process all PDFs: validate and deduplicate"""
        logger.info("Starting PDF validation and deduplication")
        logger.info("Using multi-method deduplication: hash, DOI, text similarity")
        
        # Collect all PDF files
        all_pdfs = []
        for source_dir in self.source_dirs:
            if os.path.exists(source_dir):
                pdfs = list(Path(source_dir).glob("*.pdf"))
                all_pdfs.extend(pdfs)
                logger.info(f"Found {len(pdfs)} PDFs in {source_dir}")
        
        self.stats['total_pdfs'] = len(all_pdfs)
        logger.info(f"Total PDFs to process: {len(all_pdfs)}")
        
        if not all_pdfs:
            logger.warning("No PDFs found to process")
            return
        
        # Step 1: Validate all PDFs
        logger.info("Step 1: Validating PDFs...")
        valid_pdfs = []
        invalid_pdfs = []
        
        for i, pdf_path in enumerate(all_pdfs, 1):
            logger.info(f"Validating {i}/{len(all_pdfs)}: {pdf_path.name}")
            
            validation_result = self.validate_pdf(pdf_path)
            
            if validation_result['is_valid']:
                valid_pdfs.append(validation_result)
                logger.info(f"  ✓ Valid: {validation_result['page_count']} pages, "
                          f"{validation_result['file_size_bytes']:,} bytes")
                
                # Log if DOI extracted
                if validation_result['extracted_doi']:
                    logger.info(f"    DOI found: {validation_result['extracted_doi']}")
            else:
                invalid_pdfs.append(validation_result)
                logger.warning(f"  ✗ Invalid: {validation_result['error']}")
                
                # Move invalid PDF
                invalid_dest = Path(self.invalid_dir) / pdf_path.name
                try:
                    shutil.move(str(pdf_path), str(invalid_dest))
                    logger.info(f"  Moved to invalid: {invalid_dest.name}")
                except Exception as e:
                    logger.error(f"  Failed to move invalid PDF: {e}")
        
        self.stats['valid_pdfs'] = len(valid_pdfs)
        self.stats['invalid_pdfs'] = len(invalid_pdfs)
        
        logger.info(f"Validation complete: {len(valid_pdfs)} valid, {len(invalid_pdfs)} invalid")
        
        # Step 2: Group and deduplicate
        logger.info("Step 2: Grouping and deduplicating...")
        pdf_groups = self.group_pdfs_by_identity(valid_pdfs)
        
        logger.info(f"Found {len(pdf_groups)} unique publication groups")
        
        final_pdfs = []
        duplicates_removed = 0
        
        for group_key, pdf_group in pdf_groups.items():
            if len(pdf_group) > 1:
                logger.info(f"Processing duplicate group '{group_key}' with {len(pdf_group)} PDFs:")
                
                for pdf in pdf_group:
                    logger.info(f"  - {pdf['filename']}: {pdf['page_count']} pages, "
                              f"{pdf['file_size_bytes']:,} bytes")
                
                best_pdf, duplicates = self.select_best_pdf(pdf_group)
                
                logger.info(f"  KEEPING: {best_pdf['filename']}")
                final_pdfs.append(best_pdf)
                
                # Move duplicates
                for dup_pdf in duplicates:
                    try:
                        src_path = Path(dup_pdf['filepath'])
                        if src_path.exists():  # Check if still exists
                            dup_dest = Path(self.duplicates_dir) / src_path.name
                            
                            # Handle filename conflicts
                            if dup_dest.exists():
                                base = dup_dest.stem
                                ext = dup_dest.suffix
                                counter = 1
                                while dup_dest.exists():
                                    dup_dest = Path(self.duplicates_dir) / f"{base}_dup{counter}{ext}"
                                    counter += 1
                            
                            shutil.move(str(src_path), str(dup_dest))
                            logger.info(f"  MOVED: {src_path.name} to duplicates")
                            duplicates_removed += 1
                    except Exception as e:
                        logger.error(f"  Failed to move duplicate: {e}")
            else:
                # No duplicates, keep the single PDF
                final_pdfs.append(pdf_group[0])
        
        self.stats['duplicates_removed'] = duplicates_removed
        self.stats['final_unique_pdfs'] = len(final_pdfs)
        
        # Step 3: Move valid unique PDFs to final directory
        logger.info("Step 3: Moving valid unique PDFs to final directory...")
        for pdf_info in final_pdfs:
            try:
                src_path = Path(pdf_info['filepath'])
                if src_path.exists():  # Only move if still exists (not already moved)
                    final_dest = Path(self.valid_dir) / src_path.name
                    
                    # Handle filename conflicts
                    if final_dest.exists() and src_path != final_dest:
                        base = final_dest.stem
                        ext = final_dest.suffix
                        counter = 1
                        while final_dest.exists():
                            final_dest = Path(self.valid_dir) / f"{base}_v{counter}{ext}"
                            counter += 1
                    
                    if src_path != final_dest:  # Don't move if already in place
                        shutil.move(str(src_path), str(final_dest))
                        logger.info(f"Moved to valid: {final_dest.name}")
            except Exception as e:
                logger.error(f"Failed to move valid PDF: {e}")

    def generate_report(self):
        """Generate final validation report"""
        logger.info("Generating validation report...")
        
        report_file = os.path.join(self.output_dir, 'validation_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("PDF Validation and Deduplication Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUMMARY:\n")
            for key, value in self.stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
            
            f.write(f"\nDEDUPLICATION METHODS:\n")
            f.write(f"Hash duplicates found: {self.stats['hash_duplicates']}\n")
            f.write(f"DOI duplicates found: {self.stats['doi_duplicates']}\n")
            f.write(f"Text similarity duplicates found: {self.stats['text_duplicates']}\n")
            
            f.write(f"\nFINAL DIRECTORIES:\n")
            f.write(f"Valid PDFs: {self.valid_dir}\n")
            f.write(f"Invalid PDFs: {self.invalid_dir}\n")
            f.write(f"Duplicate PDFs: {self.duplicates_dir}\n")
            
            if self.stats['total_pdfs'] > 0:
                success_rate = (self.stats['final_unique_pdfs'] / self.stats['total_pdfs']) * 100
                f.write(f"\nFinal Success Rate: {success_rate:.1f}%\n")
        
        logger.info(f"Report saved to: {report_file}")

    def print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*60)
        logger.info("PDF VALIDATION & DEDUPLICATION SUMMARY")
        logger.info("="*60)
        
        for key, value in self.stats.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value:,}")
        
        if self.stats['total_pdfs'] > 0:
            success_rate = (self.stats['final_unique_pdfs'] / self.stats['total_pdfs']) * 100
            logger.info(f"Final Success Rate: {success_rate:.1f}%")
        
        logger.info(f"\nDEDUPLICATION BREAKDOWN:")
        logger.info(f"🔍 Hash-based (exact): {self.stats['hash_duplicates']} duplicates")
        logger.info(f"📄 DOI-based: {self.stats['doi_duplicates']} duplicates")
        logger.info(f"📝 Text similarity: {self.stats['text_duplicates']} duplicates")
        
        logger.info(f"\nFINAL DIRECTORIES:")
        logger.info(f"✓ Valid PDFs: {self.valid_dir}")
        logger.info(f"✗ Invalid PDFs: {self.invalid_dir}")
        logger.info(f"📋 Duplicates: {self.duplicates_dir}")
        
        if self.stats['duplicates_removed'] > 0:
            logger.info(f"\n🔄 Removed {self.stats['duplicates_removed']} duplicate PDFs")
        
        logger.info(f"\n🎯 FINAL RESULT: {self.stats['final_unique_pdfs']} unique, valid PDFs ready for use!")

def main():
    validator = PDFValidator()
    validator.process_all_pdfs()
    validator.generate_report()
    validator.print_summary()

if __name__ == "__main__":
    main()

print("✅ Enhanced PDF validation and deduplication module loaded successfully")
#!/usr/bin/env python3
"""
Stage 4: PDF Validation and Deduplication
Validates all downloaded PDFs and removes duplicates, keeping the best quality version
"""

import os
import csv
import shutil
import logging
import PyPDF2
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFValidator:
    def __init__(self):
        self.input_file = "/app/data/publications.txt"
        self.source_dirs = [
            "/app/output/openalex",
            "/app/output/core", 
            "/app/output/other"
        ]
        self.valid_dir = "/app/output/valid_pdfs"
        self.invalid_dir = "/app/output/invalid_pdfs" 
        self.duplicates_dir = "/app/output/duplicate_pdfs"
        self.output_dir = "/app/output"
        
        # Create output directories
        for dir_path in [self.valid_dir, self.invalid_dir, self.duplicates_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Load publication data for DOI mapping
        self.publications = self.load_publications()
        self.doi_to_pubid = self.create_doi_mapping()
        
        self.stats = {
            'total_pdfs': 0,
            'valid_pdfs': 0,
            'invalid_pdfs': 0,
            'duplicates_removed': 0,
            'final_unique_pdfs': 0
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

    def create_doi_mapping(self):
        """Create DOI to pub_id mapping for deduplication"""
        doi_mapping = {}
        for pub_id, pub_data in self.publications.items():
            doi = pub_data.get('doi', '').strip()
            if doi and doi.lower() != 'nan':
                # Normalize DOI
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
        """Validate a single PDF file"""
        result = {
            'filepath': str(pdf_path),
            'filename': pdf_path.name,
            'is_valid': False,
            'page_count': 0,
            'file_size_bytes': 0,
            'error': None
        }
        
        try:
            # Check file size
            file_size = pdf_path.stat().st_size
            result['file_size_bytes'] = file_size
            
            if file_size < 1000:  # Less than 1KB
                result['error'] = 'File too small (< 1KB)'
                return result
            
            # Try to open with PyPDF2
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Check if encrypted
                if pdf_reader.is_encrypted:
                    result['error'] = 'PDF is encrypted'
                    return result
                
                # Get page count
                page_count = len(pdf_reader.pages)
                result['page_count'] = page_count
                
                if page_count == 0:
                    result['error'] = 'PDF has 0 pages'
                    return result
                
                if page_count < 2:
                    result['error'] = f'PDF too short ({page_count} page)'
                    return result
                
                # Try to extract text from first page
                try:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    if not text or len(text.strip()) < 10:
                        result['error'] = 'Cannot extract meaningful text'
                        return result
                except:
                    result['error'] = 'Cannot read PDF content'
                    return result
                
                result['is_valid'] = True
                
        except PyPDF2.errors.PdfReadError as e:
            result['error'] = f'PDF read error: {str(e)}'
        except Exception as e:
            result['error'] = f'Validation error: {str(e)}'
        
        return result

    def extract_metadata_from_filename(self, filename):
        """Extract DOI and other metadata from consistent filename format"""
        # Expected format: {doi}_{title_words}_{year}.pdf
        name_without_ext = filename.replace('.pdf', '')
        parts = name_without_ext.split('_')
        
        if len(parts) < 3:
            return None
        
        # First part should be DOI (normalized)
        doi_part = parts[0]
        if doi_part == 'no_doi':
            return None
        
        # Reconstruct DOI with proper format
        if '/' in doi_part:
            reconstructed_doi = doi_part
        else:
            # This might need manual inspection - log for review
            logger.debug(f"Unusual DOI format in filename: {doi_part}")
            reconstructed_doi = doi_part
        
        return {
            'doi': reconstructed_doi,
            'filename_parts': parts
        }

    def group_pdfs_by_identity(self, valid_pdfs):
        """Group PDFs by their publication identity (DOI or filename similarity)"""
        groups = defaultdict(list)
        
        for pdf_info in valid_pdfs:
            filename = pdf_info['filename']
            
            # Try to extract DOI from filename
            metadata = self.extract_metadata_from_filename(filename)
            
            if metadata and metadata['doi']:
                # Group by DOI
                normalized_doi = self.normalize_doi(metadata['doi'])
                if normalized_doi in self.doi_to_pubid:
                    group_key = f"doi_{normalized_doi}"
                    groups[group_key].append(pdf_info)
                    continue
            
            # If no DOI match, group by filename pattern (first part before first underscore)
            # This catches cases where the same pub_id was downloaded from different sources
            filename_base = filename.split('_')[0] if '_' in filename else filename
            group_key = f"filename_{filename_base}"
            groups[group_key].append(pdf_info)
        
        return groups

    def select_best_pdf(self, pdf_group):
        """Select the best PDF from a group of duplicates"""
        if len(pdf_group) == 1:
            return pdf_group[0], []
        
        # Sort by: 1) Page count (more is better), 2) File size (larger is better)
        sorted_pdfs = sorted(
            pdf_group, 
            key=lambda x: (x['page_count'], x['file_size_bytes']), 
            reverse=True
        )
        
        best_pdf = sorted_pdfs[0]
        duplicates = sorted_pdfs[1:]
        
        return best_pdf, duplicates

    def process_all_pdfs(self):
        """Process all PDFs: validate and deduplicate"""
        logger.info("Starting PDF validation and deduplication")
        
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
                        dup_dest = Path(self.duplicates_dir) / src_path.name
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
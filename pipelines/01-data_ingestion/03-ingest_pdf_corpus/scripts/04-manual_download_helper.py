#!/usr/bin/env python3
# ============================================================================
# FILE: scripts/05-manual_download_helper.py
# LOCATION: 01-data_ingestion/03-ingest_pdf_corpus/scripts/05-manual_download_helper.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 03
# PURPOSE: Docker-compatible CLI tool to help manually download PDFs
# ============================================================================

"""
MODULE OVERVIEW:
This module provides a Docker-compatible CLI interface for manually downloading PDFs
that couldn't be automatically fetched. It generates URL lists for manual opening,
tracks progress including rename status, and helps rename files to match the pipeline's 
naming convention.

CLASSES:
- ManualDownloadHelper: Main class for manual download assistance

METHODS:
- __init__(): Initializes the helper with file paths and creates directories
- load_manual_downloads(): Reads the manual_download.txt file
- load_progress(): Loads previous session progress
- save_progress(): Saves current session progress including rename status
- normalize_text(): Cleans text for consistent filename generation
- create_filename(): Generates standardized PDF filename from metadata
- display_urls_for_batch(): Shows URLs for manual copying (Docker-compatible)
- scan_manual_folder(): Scans for new PDFs in manual folder
- rename_downloaded_files(): Renames manually downloaded files to standard format
- interactive_download_session(): Main interactive CLI loop
- print_summary(): Displays session statistics

HYPERPARAMETERS:
- BATCH_SIZE: 10 (number of URLs to display at once)
- WAIT_TIME: User-controlled (waits for manual confirmation)

DEPENDENCIES:
- json: For progress tracking
- pathlib: For file operations
- time: For delays
- csv: For reading manual download list
- re: For text normalization
"""

import os
import csv
import json
import time
import re
from pathlib import Path
from datetime import datetime

class ManualDownloadHelper:
    def __init__(self):
        # File paths
        self.manual_list_file = Path("/app/output/manual_download.txt")
        self.manual_download_dir = Path("/app/output/manual")
        self.progress_file = Path("/app/output/manual_download_progress.json")
        self.renamed_dir = Path("/app/output/manual_renamed")
        
        # Create directories
        self.manual_download_dir.mkdir(parents=True, exist_ok=True)
        self.renamed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.downloads = self.load_manual_downloads()
        self.progress = self.load_progress()
        
        # Track processed files to avoid re-processing
        self.processed_files = set()
        if 'processed_files' in self.progress:
            self.processed_files = set(self.progress['processed_files'])
        
        # Stats
        self.stats = {
            'total': len(self.downloads),
            'downloaded': 0,
            'renamed': 0,
            'skipped': 0,
            'pending': 0
        }
        
        # Calculate initial stats from progress
        self.update_stats_from_progress()

    def load_manual_downloads(self):
        """Load the list of URLs that need manual download"""
        downloads = []
        if not self.manual_list_file.exists():
            print(f"❌ No manual download file found at {self.manual_list_file}")
            return downloads
        
        with open(self.manual_list_file, 'r', encoding='utf-8') as f:
            # Skip header if exists
            first_line = f.readline()
            if 'pub_id' in first_line.lower():
                # Has header, parse as TSV
                f.seek(0)
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    downloads.append({
                        'pub_id': row.get('pub_id', ''),
                        'doi': row.get('doi', ''),
                        'title': row.get('title', ''),
                        'url': row.get('url', ''),
                        'source': row.get('source', ''),
                        'status': 'pending'  # pending, downloaded, renamed, skipped
                    })
            else:
                # No header, just URLs
                f.seek(0)
                for line in f:
                    url = line.strip()
                    if url:
                        downloads.append({
                            'pub_id': f'manual_{len(downloads)+1}',
                            'doi': '',
                            'title': '',
                            'url': url,
                            'source': 'manual',
                            'status': 'pending'
                        })
        
        print(f"📋 Loaded {len(downloads)} URLs for manual download")
        return downloads

    def load_progress(self):
        """Load progress from previous session"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                print(f"📂 Loaded progress from previous session")
                return progress
        return {'publications': {}, 'processed_files': []}

    def save_progress(self):
        """Save current progress including rename status"""
        # Update processed files list
        self.progress['processed_files'] = list(self.processed_files)
        
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        print(f"💾 Progress saved to {self.progress_file}")

    def update_stats_from_progress(self):
        """Update statistics based on loaded progress"""
        # Reset stats
        self.stats = {
            'total': len(self.downloads),
            'downloaded': 0,
            'renamed': 0,
            'skipped': 0,
            'pending': 0
        }
        
        # Count each status
        for download in self.downloads:
            pub_id = download['pub_id']
            if pub_id in self.progress.get('publications', {}):
                status = self.progress['publications'][pub_id]
                if status == 'renamed':
                    self.stats['renamed'] += 1
                elif status == 'downloaded':
                    self.stats['downloaded'] += 1
                elif status == 'skipped':
                    self.stats['skipped'] += 1
                else:
                    self.stats['pending'] += 1
            else:
                self.stats['pending'] += 1

    def normalize_text(self, text):
        """Normalize text for consistent naming"""
        if not text:
            return ""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'\s+', '_', text.strip())
        return text

    def create_filename(self, download_info):
        """Create standardized filename from download info"""
        # Try to match the standard format: {doi}_{title_words}_{year}.pdf
        
        # Clean DOI
        doi = download_info.get('doi', '').lower()
        if doi and doi != 'no_doi':
            doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            doi = doi.replace('doi:', '').strip('/')
            doi = self.normalize_text(doi) if doi else 'no_doi'
        else:
            doi = f"manual_{download_info['pub_id']}"
        
        # Get first 5 words of title
        title = download_info.get('title', '')
        if title:
            title_words = self.normalize_text(title).split('_')[:5]
            title_part = '_'.join(title_words)
        else:
            title_part = 'no_title'
        
        # Create filename (no year since we don't have it in manual_download.txt)
        filename = f"{doi}_{title_part}.pdf"
        
        # Ensure filename isn't too long
        if len(filename) > 200:
            filename = filename[:196] + ".pdf"
        
        return filename

    def display_urls_for_batch(self, urls, batch_info, start_idx):
        """Display URLs for manual copying (Docker-compatible version)"""
        print(f"\n" + "="*60)
        print(f"📋 MANUAL DOWNLOAD BATCH")
        print("="*60)
        print(f"Opening batch with {len(urls)} URLs for manual download:")
        print("")
        
        # Get ALL pending downloads for HTML generation
        all_pending = []
        for download in self.downloads:
            pub_id = download['pub_id']
            status = self.progress.get('publications', {}).get(pub_id, 'pending')
            if status not in ['renamed', 'skipped']:
                all_pending.append(download)
        
        # Create an enhanced HTML file with batch navigation
        html_file = Path("/app/output/batch_urls.html")
        txt_file = Path("/app/output/batch_urls.txt")
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>PDF Download Manager</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .instructions { background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #ffc107; }
        .controls { background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 8px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        .batch-nav { 
            background: #f5f5f5; padding: 10px 15px; border-radius: 8px; 
            display: flex; gap: 15px; align-items: center; margin: 15px 0;
            justify-content: space-between;
        }
        .batch-info { font-size: 16px; font-weight: bold; color: #333; }
        .nav-btn { 
            background: #2196F3; color: white; padding: 8px 20px; 
            border: none; border-radius: 4px; cursor: pointer; font-size: 14px;
            min-width: 100px;
        }
        .nav-btn:hover { background: #1976d2; }
        .nav-btn:disabled { background: #ccc; cursor: not-allowed; }
        .link-item { 
            margin: 15px 0; padding: 15px; background: #fff; border: 2px solid #e0e0e0; 
            border-radius: 8px; transition: all 0.3s; 
        }
        .link-item:hover { border-color: #4CAF50; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .link-item.downloaded { background: #e8f5e9; border-color: #4CAF50; }
        .pub-info { font-weight: bold; color: #1976d2; margin-bottom: 8px; font-size: 16px; }
        .title { color: #666; margin: 5px 0; font-size: 14px; }
        .url-container { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
        .url { flex: 1; padding: 8px; background: #f5f5f5; border-radius: 4px; font-size: 12px; word-break: break-all; }
        .download-btn { 
            background: #2196F3; color: white; padding: 8px 16px; 
            border: none; border-radius: 4px; cursor: pointer; text-decoration: none;
            display: inline-block; font-size: 14px;
        }
        .download-btn:hover { background: #1976d2; }
        .download-btn.done { background: #4CAF50; }
        button { 
            background: #4CAF50; color: white; padding: 12px 24px; 
            border: none; border-radius: 4px; cursor: pointer;
            font-size: 16px; font-weight: bold;
        }
        button:hover { background: #45a049; }
        button.secondary { background: #ff9800; }
        button.secondary:hover { background: #f57c00; }
        .checkbox { width: 20px; height: 20px; margin-right: 10px; cursor: pointer; }
        .stats { 
            background: #f5f5f5; padding: 10px; border-radius: 4px; 
            text-align: center; font-size: 18px; margin: 20px 0;
        }
        .progress-bar { 
            width: 100%; height: 30px; background: #e0e0e0; 
            border-radius: 15px; overflow: hidden; margin: 20px 0;
        }
        .progress-fill { 
            height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); 
            transition: width 0.3s; display: flex; align-items: center; 
            justify-content: center; color: white; font-weight: bold;
        }
        .warning { color: #ff5722; font-weight: bold; }
        .batch-container { display: none; }
        .batch-container.active { display: block; }
        .hidden { display: none; }
    </style>
    <script>
        let currentBatch = 0;
        let totalBatches = 0;
        let batchSize = 10;
        let allUrls = [];
        let downloadStatus = {};
        
        function initializeBatches() {
            const containers = document.querySelectorAll('.batch-container');
            totalBatches = containers.length;
            
            // Initialize download status
            document.querySelectorAll('.link-item').forEach((item) => {
                const id = item.dataset.pubid;
                if (!downloadStatus[id]) {
                    downloadStatus[id] = false;
                }
            });
            
            // Restore any previous marks
            loadProgress();
            
            showBatch(0);
            updateStats();
        }
        
        function showBatch(batchNum) {
            if (batchNum < 0 || batchNum >= totalBatches) return;
            
            // Hide all batches
            document.querySelectorAll('.batch-container').forEach(container => {
                container.classList.remove('active');
            });
            
            // Show selected batch
            document.getElementById('batch-' + batchNum).classList.add('active');
            currentBatch = batchNum;
            
            // Update navigation
            document.getElementById('batch-info').textContent = 
                'Batch ' + (batchNum + 1) + ' of ' + totalBatches + 
                ' (Items ' + (batchNum * batchSize + 1) + '-' + 
                Math.min((batchNum + 1) * batchSize, document.querySelectorAll('.link-item').length) + ')';
            
            // Enable/disable nav buttons
            document.getElementById('prev-btn').disabled = (batchNum === 0);
            document.getElementById('next-btn').disabled = (batchNum === totalBatches - 1);
            
            updateBatchStats();
        }
        
        function nextBatch() {
            if (currentBatch < totalBatches - 1) {
                showBatch(currentBatch + 1);
            }
        }
        
        function prevBatch() {
            if (currentBatch > 0) {
                showBatch(currentBatch - 1);
            }
        }
        
        function markDownloaded(pubId) {
            const item = document.querySelector('[data-pubid="' + pubId + '"]');
            const btn = document.getElementById('btn-' + pubId);
            const checkbox = document.getElementById('check-' + pubId);
            
            if (!downloadStatus[pubId]) {
                item.classList.add('downloaded');
                btn.classList.add('done');
                btn.textContent = '✓ Downloaded';
                checkbox.checked = true;
                downloadStatus[pubId] = true;
                saveProgress();
                updateStats();
            }
        }
        
        function updateStats() {
            const total = Object.keys(downloadStatus).length;
            const downloaded = Object.values(downloadStatus).filter(v => v).length;
            const percent = total > 0 ? Math.round((downloaded / total) * 100) : 0;
            
            document.getElementById('progress-fill').style.width = percent + '%';
            document.getElementById('progress-text').textContent = percent + '%';
            document.getElementById('stats').textContent = downloaded + ' / ' + total + ' PDFs downloaded';
            
            updateBatchStats();
        }
        
        function updateBatchStats() {
            const currentBatchItems = document.querySelectorAll('#batch-' + currentBatch + ' .link-item');
            let batchDownloaded = 0;
            currentBatchItems.forEach(item => {
                if (downloadStatus[item.dataset.pubid]) batchDownloaded++;
            });
            
            document.getElementById('batch-stats').textContent = 
                'This batch: ' + batchDownloaded + ' / ' + currentBatchItems.length + ' downloaded';
        }
        
        function openBatchLinks() {
            const currentBatchItems = document.querySelectorAll('#batch-' + currentBatch + ' .link-item');
            const toOpen = [];
            
            currentBatchItems.forEach(item => {
                if (!downloadStatus[item.dataset.pubid]) {
                    const link = item.querySelector('.pdf-link');
                    if (link) toOpen.push(link.href);
                }
            });
            
            if (toOpen.length === 0) {
                alert('All PDFs in this batch have been downloaded!');
                return;
            }
            
            if (confirm('Open ' + toOpen.length + ' tabs from this batch?')) {
                toOpen.forEach(url => window.open(url, '_blank'));
                setTimeout(() => {
                    alert('Tabs opened! Click "Mark Downloaded" buttons after saving PDFs.');
                }, 1000);
            }
        }
        
        function openSelected() {
            const num = parseInt(prompt('How many tabs to open?', '5')) || 5;
            const currentBatchItems = document.querySelectorAll('#batch-' + currentBatch + ' .link-item');
            const toOpen = [];
            
            currentBatchItems.forEach(item => {
                if (!downloadStatus[item.dataset.pubid] && toOpen.length < num) {
                    const link = item.querySelector('.pdf-link');
                    if (link) toOpen.push(link.href);
                }
            });
            
            toOpen.forEach(url => window.open(url, '_blank'));
            alert('Opened ' + toOpen.length + ' tabs.');
        }
        
        function markBatchDownloaded() {
            if (confirm('Mark all PDFs in this batch as downloaded?')) {
                const currentBatchItems = document.querySelectorAll('#batch-' + currentBatch + ' .link-item');
                currentBatchItems.forEach(item => {
                    markDownloaded(item.dataset.pubid);
                });
            }
        }
        
        function saveProgress() {
            localStorage.setItem('pdfDownloadStatus', JSON.stringify(downloadStatus));
        }
        
        function loadProgress() {
            const saved = localStorage.getItem('pdfDownloadStatus');
            if (saved) {
                const savedStatus = JSON.parse(saved);
                Object.keys(savedStatus).forEach(pubId => {
                    if (savedStatus[pubId]) {
                        downloadStatus[pubId] = true;
                        const item = document.querySelector('[data-pubid="' + pubId + '"]');
                        if (item) {
                            item.classList.add('downloaded');
                            const btn = document.getElementById('btn-' + pubId);
                            if (btn) {
                                btn.classList.add('done');
                                btn.textContent = '✓ Downloaded';
                            }
                            const checkbox = document.getElementById('check-' + pubId);
                            if (checkbox) checkbox.checked = true;
                        }
                    }
                });
            }
        }
        
        function resetProgress() {
            if (confirm('Reset all download marks?')) {
                localStorage.removeItem('pdfDownloadStatus');
                location.reload();
            }
        }
        
        window.onload = function() {
            initializeBatches();
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>📥 PDF Download Manager</h1>
        
        <div class="instructions">
            <strong>⚠️ IMPORTANT - Set Chrome Download Location First:</strong><br>
            1. Go to Chrome Settings → Downloads<br>
            2. Change location to: <code>./output/manual/</code><br>
            3. (Optional) Turn OFF "Ask where to save each file" for faster downloads<br>
            4. Click buttons below to open PDFs, they'll auto-save to the right folder!
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%">
                <span id="progress-text">0%</span>
            </div>
        </div>
        
        <div class="stats" id="stats">0 / 0 PDFs downloaded</div>
        
        <div class="batch-nav">
            <button class="nav-btn" id="prev-btn" onclick="prevBatch()">← Previous 10</button>
            <div style="text-align: center;">
                <div class="batch-info" id="batch-info">Batch 1 of 1</div>
                <div style="color: #666; font-size: 14px;" id="batch-stats">This batch: 0 / 0 downloaded</div>
            </div>
            <button class="nav-btn" id="next-btn" onclick="nextBatch()">Next 10 →</button>
        </div>
        
        <div class="controls">
            <button onclick="openBatchLinks()">🚀 Open This Batch</button>
            <button onclick="openSelected()" class="secondary">📂 Open First N</button>
            <button onclick="markBatchDownloaded()" style="background: #4CAF50;">✓ Mark Batch Downloaded</button>
            <button onclick="resetProgress()" style="background: #f44336;">↻ Reset All</button>
        </div>
        
        <hr>
"""
        
        # Generate batches of 10
        batch_num = 0
        for batch_start in range(0, len(all_pending), 10):
            batch_end = min(batch_start + 10, len(all_pending))
            batch_items = all_pending[batch_start:batch_end]
            
            html_content += f'<div class="batch-container" id="batch-{batch_num}">\n'
            html_content += f'<h2>Batch {batch_num + 1} - Items {batch_start + 1} to {batch_end}:</h2>\n'
            
            for i, download in enumerate(batch_items, start=batch_start):
                pub_id = download['pub_id']
                url = download['url']
                title = download.get('title', 'No title')[:100]
                
                html_content += f"""
        <div class="link-item" id="item-{pub_id}" data-pubid="{pub_id}">
            <input type="checkbox" class="checkbox" id="check-{pub_id}" 
                   onchange="if(this.checked) markDownloaded('{pub_id}')" />
            <div style="display: inline-block; flex: 1;">
                <div class="pub-info">#{i+1} - [{pub_id}]</div>
                <div class="title">Title: {title}...</div>
                <div class="url-container">
                    <div class="url">{url}</div>
                    <a href="{url}" target="_blank" class="download-btn pdf-link" 
                       onclick="setTimeout(() => markDownloaded('{pub_id}'), 1000); return true;">
                        Open PDF
                    </a>
                    <button class="download-btn" id="btn-{pub_id}" onclick="markDownloaded('{pub_id}')">
                        Mark Downloaded
                    </button>
                </div>
            </div>
        </div>
"""
            
            html_content += '</div>\n'  # Close batch container
            batch_num += 1
        
        html_content += f"""
        <hr>
        <div class="batch-nav" style="margin-top: 30px;">
            <button class="nav-btn" onclick="prevBatch()">← Previous 10</button>
            <div class="batch-info">Navigate between batches using buttons</div>
            <button class="nav-btn" onclick="nextBatch()">Next 10 →</button>
        </div>
        <p style="text-align: center; color: #666;">
            Total pending URLs: {len(all_pending)} | 
            <span class="warning">Remember to set Chrome's download folder first!</span>
        </p>
    </div>
</body>
</html>"""
        
        # Write HTML file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Write current batch URLs to text file
        txt_content = [d['url'] for d in batch_info if self.progress.get('publications', {}).get(d['pub_id'], 'pending') not in ['renamed', 'skipped']]
        with open(txt_file, 'w') as f:
            f.write('\n'.join(txt_content))
        
        # Terminal output
        for i, info in enumerate(batch_info, start=start_idx+1):
            status = self.progress.get('publications', {}).get(info['pub_id'], 'pending')
            status_emoji = "✅" if status == 'renamed' else "📥" if status == 'downloaded' else "⏳"
            print(f"{i}. {status_emoji} [{info['pub_id']}]")
            print(f"   Title: {info['title'][:60] if info['title'] else 'No title'}...")
            print("")
        
        print("="*60)
        print("📌 HOW TO USE:")
        print("")
        print("1. ⚙️  Set Chrome's download folder to: ./output/manual/")
        print("")
        print("2. 🌐 Open: ./output/batch_urls.html")
        print("   • Use 'Previous 10' / 'Next 10' buttons to navigate batches")
        print("   • Click 'Open This Batch' to open current 10 URLs")
        print("   • Mark each PDF as downloaded after saving")
        print("")
        print("3. ✅ Progress is saved automatically in the browser")
        print("="*60)
        """Display URLs for manual copying (Docker-compatible version)"""
        print(f"\n" + "="*60)
        print(f"📋 MANUAL DOWNLOAD BATCH")
        print("="*60)
        print(f"Opening {len(urls)} URLs for manual download:")
        print("")
        
        # Create an enhanced HTML file with download tracking
        html_file = Path("/app/output/batch_urls.html")
        txt_file = Path("/app/output/batch_urls.txt")
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>PDF Download Manager</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .instructions { background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #ffc107; }
        .controls { background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 8px; display: flex; gap: 10px; flex-wrap: wrap; }
        .link-item { 
            margin: 15px 0; padding: 15px; background: #fff; border: 2px solid #e0e0e0; 
            border-radius: 8px; transition: all 0.3s; 
        }
        .link-item:hover { border-color: #4CAF50; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .link-item.downloaded { background: #e8f5e9; border-color: #4CAF50; }
        .pub-info { font-weight: bold; color: #1976d2; margin-bottom: 8px; font-size: 16px; }
        .title { color: #666; margin: 5px 0; font-size: 14px; }
        .url-container { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
        .url { flex: 1; padding: 8px; background: #f5f5f5; border-radius: 4px; font-size: 12px; word-break: break-all; }
        .download-btn { 
            background: #2196F3; color: white; padding: 8px 16px; 
            border: none; border-radius: 4px; cursor: pointer; text-decoration: none;
            display: inline-block; font-size: 14px;
        }
        .download-btn:hover { background: #1976d2; }
        .download-btn.done { background: #4CAF50; }
        button { 
            background: #4CAF50; color: white; padding: 12px 24px; 
            border: none; border-radius: 4px; cursor: pointer;
            font-size: 16px; font-weight: bold;
        }
        button:hover { background: #45a049; }
        button.secondary { background: #ff9800; }
        button.secondary:hover { background: #f57c00; }
        .checkbox { width: 20px; height: 20px; margin-right: 10px; cursor: pointer; }
        .stats { 
            background: #f5f5f5; padding: 10px; border-radius: 4px; 
            text-align: center; font-size: 18px; margin: 20px 0;
        }
        .progress-bar { 
            width: 100%; height: 30px; background: #e0e0e0; 
            border-radius: 15px; overflow: hidden; margin: 20px 0;
        }
        .progress-fill { 
            height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); 
            transition: width 0.3s; display: flex; align-items: center; 
            justify-content: center; color: white; font-weight: bold;
        }
        .warning { color: #ff5722; font-weight: bold; }
    </style>
    <script>
        let totalLinks = 0;
        let downloadedCount = 0;
        
        function markDownloaded(index) {
            const item = document.getElementById('item-' + index);
            const btn = document.getElementById('btn-' + index);
            const checkbox = document.getElementById('check-' + index);
            
            if (!item.classList.contains('downloaded')) {
                item.classList.add('downloaded');
                btn.classList.add('done');
                btn.textContent = '✓ Downloaded';
                checkbox.checked = true;
                downloadedCount++;
                updateProgress();
            }
        }
        
        function updateProgress() {
            const percent = Math.round((downloadedCount / totalLinks) * 100);
            document.getElementById('progress-fill').style.width = percent + '%';
            document.getElementById('progress-text').textContent = percent + '%';
            document.getElementById('stats').textContent = downloadedCount + ' / ' + totalLinks + ' PDFs downloaded';
            
            if (downloadedCount === totalLinks) {
                alert('All PDFs downloaded! Return to terminal to rename them.');
            }
        }
        
        function openAllLinks() {
            const checkboxes = document.querySelectorAll('.checkbox:not(:checked)');
            const count = checkboxes.length;
            
            if (count === 0) {
                alert('All PDFs have been downloaded!');
                return;
            }
            
            if (confirm('This will open ' + count + ' tabs for PDFs not yet downloaded. Continue?')) {
                checkboxes.forEach(cb => {
                    const index = cb.dataset.index;
                    const link = document.getElementById('link-' + index);
                    window.open(link.href, '_blank');
                });
                
                setTimeout(() => {
                    alert('Tabs opened! After downloading, click the "Downloaded" buttons to track progress.');
                }, 1000);
            }
        }
        
        function openSelected() {
            const checkboxes = document.querySelectorAll('.checkbox:not(:checked)');
            let urls = [];
            checkboxes.forEach(cb => {
                const index = cb.dataset.index;
                const link = document.getElementById('link-' + index);
                urls.push(link.href);
            });
            
            if (urls.length === 0) {
                alert('All PDFs have been marked as downloaded!');
                return;
            }
            
            const batchSize = prompt('How many tabs to open? (default: 5)', '5');
            const num = parseInt(batchSize) || 5;
            const toOpen = urls.slice(0, num);
            
            toOpen.forEach(url => window.open(url, '_blank'));
            alert('Opened ' + toOpen.length + ' tabs. Click "Downloaded" buttons after saving PDFs.');
        }
        
        function markAllDownloaded() {
            if (confirm('Mark ALL PDFs as downloaded?')) {
                document.querySelectorAll('.link-item').forEach((item, i) => {
                    markDownloaded(i);
                });
            }
        }
        
        function resetAll() {
            if (confirm('Reset all download marks?')) {
                document.querySelectorAll('.link-item').forEach(item => {
                    item.classList.remove('downloaded');
                });
                document.querySelectorAll('.download-btn').forEach(btn => {
                    btn.classList.remove('done');
                    btn.textContent = 'Mark Downloaded';
                });
                document.querySelectorAll('.checkbox').forEach(cb => {
                    cb.checked = false;
                });
                downloadedCount = 0;
                updateProgress();
            }
        }
        
        function copyUndownloaded() {
            const urls = [];
            document.querySelectorAll('.checkbox:not(:checked)').forEach(cb => {
                const index = cb.dataset.index;
                const link = document.getElementById('link-' + index);
                urls.push(link.href);
            });
            
            if (urls.length === 0) {
                alert('No URLs to copy - all marked as downloaded!');
                return;
            }
            
            navigator.clipboard.writeText(urls.join('\\n'));
            alert('Copied ' + urls.length + ' undownloaded URLs to clipboard!');
        }
        
        window.onload = function() {
            totalLinks = document.querySelectorAll('.link-item').length;
            updateProgress();
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>📥 PDF Download Manager</h1>
        
        <div class="instructions">
            <strong>⚠️ IMPORTANT - Set Chrome Download Location First:</strong><br>
            1. Go to Chrome Settings → Downloads<br>
            2. Change location to: <code>./output/manual/</code><br>
            3. (Optional) Turn OFF "Ask where to save each file" for faster downloads<br>
            4. Click buttons below to open PDFs, they'll auto-save to the right folder!
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%">
                <span id="progress-text">0%</span>
            </div>
        </div>
        
        <div class="stats" id="stats">0 / 0 PDFs downloaded</div>
        
        <div class="controls">
            <button onclick="openAllLinks()">🚀 Open All Remaining</button>
            <button onclick="openSelected()" class="secondary">📂 Open First N Tabs</button>
            <button onclick="markAllDownloaded()" style="background: #4CAF50;">✓ Mark All Downloaded</button>
            <button onclick="copyUndownloaded()" style="background: #2196F3;">📋 Copy Undownloaded URLs</button>
            <button onclick="resetAll()" style="background: #f44336;">↻ Reset</button>
        </div>
        
        <hr>
        <h2>PDF Links:</h2>
"""
        
        txt_content = []
        pending_count = 0
        
        for i, (url, info) in enumerate(zip(urls, batch_info)):
            # Check status
            status = self.progress.get('publications', {}).get(info['pub_id'], 'pending')
            
            if status not in ['renamed', 'skipped']:
                pending_count += 1
                
                # Add to HTML with checkbox and download tracking
                html_content += f"""
        <div class="link-item" id="item-{i}">
            <input type="checkbox" class="checkbox" id="check-{i}" data-index="{i}" onchange="if(this.checked) markDownloaded({i})" />
            <div style="display: inline-block; flex: 1;">
                <div class="pub-info">#{i+1} - [{info['pub_id']}]</div>
                <div class="title">Title: {info['title'][:100] if info['title'] else 'No title available'}...</div>
                <div class="url-container">
                    <div class="url">{url}</div>
                    <a href="{url}" target="_blank" class="download-btn" id="link-{i}" 
                       onclick="setTimeout(() => markDownloaded({i}), 1000); return true;">
                        Open PDF
                    </a>
                    <button class="download-btn" id="btn-{i}" onclick="markDownloaded({i})">
                        Mark Downloaded
                    </button>
                </div>
            </div>
        </div>
"""
                txt_content.append(url)
                
                # Print to terminal
                print(f"{i+1}. [{info['pub_id']}]")
                print(f"   Title: {info['title'][:60] if info['title'] else 'No title'}...")
                print("")
        
        html_content += f"""
        <hr>
        <p style="text-align: center; color: #666;">
            Total pending URLs: {pending_count} | 
            <span class="warning">Remember to set Chrome's download folder first!</span>
        </p>
    </div>
</body>
</html>"""
        
        # Write HTML file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Write text file
        with open(txt_file, 'w') as f:
            f.write('\n'.join(txt_content))
        
        print("="*60)
        print("📌 HOW TO USE:")
        print("")
        print("1. ⚙️  FIRST - Set Chrome's download folder:")
        print("   • Open Chrome Settings → Downloads")
        print("   • Change location to: ./output/manual/")
        print("   • Turn OFF 'Ask where to save' for auto-download")
        print("")
        print("2. 🌐 Open the download manager:")
        print("   • Open: ./output/batch_urls.html")
        print("   • Click 'Open All Remaining' or 'Open First N Tabs'")
        print("   • PDFs will auto-download to ./output/manual/")
        print("")
        print("3. ✅ Track progress in the HTML page")
        print("   • Click 'Mark Downloaded' as you save each PDF")
        print("   • Progress bar shows completion status")
        print("")
        print("="*60)

    def scan_manual_folder(self):
        """Scan manual folder for new PDF files"""
        new_files = []
        existing_files = list(self.manual_download_dir.glob("*.pdf"))
        
        for pdf_file in existing_files:
            if pdf_file.name not in self.processed_files:
                new_files.append(pdf_file)
        
        if new_files:
            print(f"\n🔍 Found {len(new_files)} new PDF(s) in manual folder:")
            for f in new_files[:5]:  # Show first 5
                print(f"   - {f.name}")
            if len(new_files) > 5:
                print(f"   ... and {len(new_files) - 5} more")
        
        return new_files

    def rename_downloaded_files(self, batch_downloads):
        """Rename downloaded files to standard format"""
        print("\n📝 Checking for downloaded files to rename...")
        
        # Scan for new files
        new_files = self.scan_manual_folder()
        
        if not new_files:
            print("No new files found to rename")
            return 0
        
        renamed_count = 0
        
        # For each new file, try to match with batch
        for pdf_path in new_files:
            print(f"\n🔍 Processing: {pdf_path.name}")
            
            # Show publications that need files
            pending_pubs = []
            for i, download in enumerate(batch_downloads):
                pub_id = download['pub_id']
                status = self.progress.get('publications', {}).get(pub_id, 'pending')
                if status not in ['renamed', 'skipped']:
                    pending_pubs.append((i, download))
            
            if not pending_pubs:
                print("All publications in this batch are already processed")
                break
            
            print("Which publication is this PDF for?")
            for idx, (i, download) in enumerate(pending_pubs, 1):
                print(f"  {idx}. [{download['pub_id']}] {download['title'][:50] if download['title'] else 'No title'}...")
            print("  s. Skip this file")
            print("  q. Done renaming")
            
            choice = input("Choice (1-" + str(len(pending_pubs)) + "/s/q): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 's':
                continue
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(pending_pubs):
                    _, selected_download = pending_pubs[idx]
                    
                    # Create new filename
                    new_filename = self.create_filename(selected_download)
                    new_path = self.renamed_dir / new_filename
                    
                    # Check if target exists
                    if new_path.exists():
                        print(f"⚠️  File already exists: {new_filename}")
                        overwrite = input("Overwrite? (y/n): ").strip().lower()
                        if overwrite != 'y':
                            continue
                    
                    # Move and rename
                    pdf_path.rename(new_path)
                    print(f"✅ Renamed to: {new_filename}")
                    renamed_count += 1
                    
                    # Update progress - mark as renamed
                    pub_id = selected_download['pub_id']
                    if 'publications' not in self.progress:
                        self.progress['publications'] = {}
                    self.progress['publications'][pub_id] = 'renamed'
                    
                    # Mark file as processed
                    self.processed_files.add(pdf_path.name)
                    
            except (ValueError, IndexError):
                print("❌ Invalid choice")
        
        if renamed_count > 0:
            self.save_progress()
        
        return renamed_count

    def interactive_download_session(self):
        """Main interactive session for manual downloads"""
        print("\n" + "="*60)
        print("📥 MANUAL PDF DOWNLOAD HELPER (Docker Version)")
        print("="*60)
        
        # Update stats first
        self.update_stats_from_progress()
        
        # Filter out already completed items
        pending_downloads = []
        for download in self.downloads:
            pub_id = download['pub_id']
            status = self.progress.get('publications', {}).get(pub_id, 'pending')
            if status not in ['renamed', 'skipped']:
                pending_downloads.append(download)
        
        print(f"\n📊 Overall Status:")
        print(f"   Total publications: {self.stats['total']}")
        print(f"   ✅ Renamed (complete): {self.stats['renamed']}")
        print(f"   📥 Downloaded (need renaming): {self.stats['downloaded']}")
        print(f"   ⏭️  Skipped: {self.stats['skipped']}")
        print(f"   ⏳ Pending: {self.stats['pending']}")
        
        # Check if there are files waiting to be renamed
        existing_manual_files = list(self.manual_download_dir.glob("*.pdf"))
        unprocessed_files = [f for f in existing_manual_files if f.name not in self.processed_files]
        
        if unprocessed_files:
            print(f"\n📁 Found {len(unprocessed_files)} PDF(s) in manual folder waiting to be renamed")
        
        if not pending_downloads and not unprocessed_files:
            print("\n✅ All URLs have been processed and renamed!")
            return
        
        batch_size = 10
        current_idx = 0
        
        # Main menu loop
        while True:
            print("\n" + "="*40)
            print("📌 MAIN MENU")
            print("="*40)
            print("What would you like to do?")
            print("  1. View/Download next batch of URLs")
            print("  2. Rename downloaded files")
            print("  3. View current statistics")
            print("  4. Skip to specific batch")
            print("  q. Save and quit")
            
            main_choice = input("\nYour choice: ").strip().lower()
            
            if main_choice == 'q':
                self.save_progress()
                print("\n👋 Session saved. You can continue later.")
                print("   Renamed files are in: /app/output/manual_renamed/")
                return
            
            elif main_choice == '1':
                # Download mode
                if not pending_downloads:
                    print("\n✅ No more URLs to download!")
                    continue
                
                # Find the next batch with pending items
                batch_found = False
                for batch_start in range(current_idx, len(pending_downloads), batch_size):
                    batch_end = min(batch_start + batch_size, len(pending_downloads))
                    batch = pending_downloads[batch_start:batch_end]
                    
                    # Check if any in this batch are actually pending
                    has_pending = False
                    for item in batch:
                        status = self.progress.get('publications', {}).get(item['pub_id'], 'pending')
                        if status == 'pending':
                            has_pending = True
                            break
                    
                    if has_pending:
                        current_idx = batch_start
                        batch_found = True
                        break
                
                if not batch_found:
                    print("\n✅ All URLs have been marked as downloaded or skipped!")
                    continue
                
                batch_end = min(current_idx + batch_size, len(pending_downloads))
                batch = pending_downloads[current_idx:batch_end]
                
                print(f"\n" + "="*40)
                print(f"📦 Download Batch {(current_idx // batch_size) + 1}")
                print(f"   Items {current_idx + 1} to {batch_end} of {len(pending_downloads)}")
                print("="*40)
                
                # Get URLs for this batch
                urls = [d['url'] for d in batch]
                
                # Display URLs for manual copying
                self.display_urls_for_batch(urls, batch, current_idx)
                
                print("\n📌 Batch Options:")
                print("   [Enter] - Continue to next batch")
                print("   [a]     - Mark all in batch as downloaded")
                print("   [s]     - Skip this entire batch")
                print("   [v]     - View URLs again")
                print("   [m]     - Return to main menu")
                
                while True:
                    choice = input("\nYour choice: ").strip().lower()
                    
                    if choice == '' or choice == 'enter':
                        current_idx = batch_end
                        break
                        
                    elif choice == 'a':
                        # Mark all as downloaded
                        for download in batch:
                            pub_id = download['pub_id']
                            current_status = self.progress.get('publications', {}).get(pub_id, 'pending')
                            if current_status == 'pending':
                                if 'publications' not in self.progress:
                                    self.progress['publications'] = {}
                                self.progress['publications'][pub_id] = 'downloaded'
                                self.stats['downloaded'] += 1
                                self.stats['pending'] -= 1
                        self.save_progress()
                        print("✅ Marked batch as downloaded")
                        current_idx = batch_end
                        break
                        
                    elif choice == 's':
                        # Skip batch
                        for download in batch:
                            pub_id = download['pub_id']
                            if 'publications' not in self.progress:
                                self.progress['publications'] = {}
                            self.progress['publications'][pub_id] = 'skipped'
                            self.stats['skipped'] += 1
                            self.stats['pending'] -= 1
                        self.save_progress()
                        current_idx = batch_end
                        print("⏭️  Skipped batch")
                        break
                        
                    elif choice == 'v':
                        # Show URLs again
                        self.display_urls_for_batch(urls, batch, current_idx)
                        
                    elif choice == 'm':
                        break
                        
                    else:
                        print("❌ Invalid choice. Please try again.")
                    
            elif main_choice == '2':
                # Rename mode
                if not unprocessed_files and self.stats['downloaded'] == 0:
                    print("\n📭 No files to rename. Download some PDFs first!")
                    continue
                
                # For renaming, use ALL pending downloads (not just current batch)
                all_pending_for_rename = []
                for download in self.downloads:
                    pub_id = download['pub_id']
                    status = self.progress.get('publications', {}).get(pub_id, 'pending')
                    if status not in ['renamed', 'skipped']:
                        all_pending_for_rename.append(download)
                
                renamed = self.rename_downloaded_files(all_pending_for_rename)
                self.stats['renamed'] += renamed
                if renamed > 0:
                    self.stats['downloaded'] = max(0, self.stats['downloaded'] - renamed)
                    self.update_stats_from_progress()
                print(f"\n✅ Renamed {renamed} files in this session")
                
            elif main_choice == '3':
                # View stats
                self.update_stats_from_progress()
                print(f"\n📊 Current Statistics:")
                print(f"   Total publications: {self.stats['total']}")
                print(f"   ✅ Renamed (complete): {self.stats['renamed']}")
                print(f"   📥 Downloaded (need renaming): {self.stats['downloaded']}")
                print(f"   ⏭️  Skipped: {self.stats['skipped']}")
                print(f"   ⏳ Pending: {self.stats['pending']}")
                
                progress_percent = (self.stats['renamed'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
                print(f"\n   Overall progress: {progress_percent:.1f}%")
                
            elif main_choice == '4':
                # Jump to specific batch
                total_batches = (len(pending_downloads) + batch_size - 1) // batch_size
                print(f"\nThere are {total_batches} batches total.")
                batch_num = input(f"Jump to batch (1-{total_batches}): ").strip()
                try:
                    batch_num = int(batch_num) - 1
                    if 0 <= batch_num < total_batches:
                        current_idx = batch_num * batch_size
                        print(f"✅ Jumped to batch {batch_num + 1}")
                    else:
                        print("❌ Invalid batch number")
                except ValueError:
                    print("❌ Please enter a number")
                    
            else:
                print("❌ Invalid choice. Please try again.")
        """Main interactive session for manual downloads"""
        print("\n" + "="*60)
        print("📥 MANUAL PDF DOWNLOAD HELPER (Docker Version)")
        print("="*60)
        
        # Filter out already completed items
        pending_downloads = []
        completed_downloads = []
        
        for download in self.downloads:
            pub_id = download['pub_id']
            status = self.progress.get('publications', {}).get(pub_id, 'pending')
            
            if status == 'renamed':
                completed_downloads.append(download)
                download['status'] = 'renamed'
            elif status == 'skipped':
                download['status'] = 'skipped'
            else:
                pending_downloads.append(download)
                download['status'] = status
        
        print(f"\n📊 Overall Status:")
        print(f"   Total publications: {len(self.downloads)}")
        print(f"   ✅ Renamed: {self.stats['renamed']}")
        print(f"   📥 Downloaded (not renamed): {self.stats['downloaded']}")
        print(f"   ⏭️  Skipped: {self.stats['skipped']}")
        print(f"   ⏳ Pending: {len(pending_downloads)}")
        
        if not pending_downloads:
            print("\n✅ All URLs have been processed and renamed!")
            return
        
        # Ask if user wants to see only pending or all
        print("\nWhat would you like to work on?")
        print("  1. Pending items only")
        print("  2. All items (including completed)")
        view_choice = input("Choice (1/2): ").strip()
        
        if view_choice == '2':
            working_list = self.downloads
        else:
            working_list = pending_downloads
        
        batch_size = 10
        current_idx = 0
        
        while current_idx < len(working_list):
            batch_end = min(current_idx + batch_size, len(working_list))
            batch = working_list[current_idx:batch_end]
            
            print(f"\n" + "="*40)
            print(f"📦 Batch {(current_idx // batch_size) + 1}")
            print(f"   Items {current_idx + 1} to {batch_end} of {len(working_list)}")
            print("="*40)
            
            # Get URLs for this batch
            urls = [d['url'] for d in batch]
            
            # Display URLs for manual copying
            self.display_urls_for_batch(urls, batch, current_idx)
            
            print("\n📌 Options:")
            print("   [Enter] - Continue to next batch")
            print("   [r]     - Rename downloaded files in this batch")
            print("   [a]     - Mark all in batch as downloaded")
            print("   [s]     - Skip this entire batch")
            print("   [v]     - View URLs again")
            print("   [q]     - Save and quit")
            
            while True:
                choice = input("\nYour choice: ").strip().lower()
                
                if choice == '' or choice == 'enter':
                    # Continue to next batch
                    current_idx = batch_end
                    break
                    
                elif choice == 'r':
                    # Rename files for this batch
                    renamed = self.rename_downloaded_files(batch)
                    self.stats['renamed'] += renamed
                    print(f"\n✅ Renamed {renamed} files")
                    # Update stats
                    self.update_stats_from_progress()
                    
                elif choice == 'a':
                    # Mark all as downloaded (but not renamed)
                    for download in batch:
                        pub_id = download['pub_id']
                        current_status = self.progress.get('publications', {}).get(pub_id, 'pending')
                        if current_status not in ['renamed', 'skipped']:
                            if 'publications' not in self.progress:
                                self.progress['publications'] = {}
                            self.progress['publications'][pub_id] = 'downloaded'
                    self.save_progress()
                    self.update_stats_from_progress()
                    print("✅ Marked batch as downloaded")
                    
                elif choice == 's':
                    # Skip batch
                    for download in batch:
                        pub_id = download['pub_id']
                        if 'publications' not in self.progress:
                            self.progress['publications'] = {}
                        self.progress['publications'][pub_id] = 'skipped'
                    self.save_progress()
                    self.update_stats_from_progress()
                    current_idx = batch_end
                    print("⏭️  Skipped batch")
                    break
                    
                elif choice == 'v':
                    # Show URLs again
                    self.display_urls_for_batch(urls, batch, current_idx)
                    
                elif choice == 'q':
                    # Quit
                    self.save_progress()
                    print("\n👋 Session saved. You can continue later.")
                    print("   Renamed files are in: /app/output/manual_renamed/")
                    return
                    
                else:
                    print("❌ Invalid choice. Please try again.")
            
            # Brief pause between batches
            if current_idx < len(working_list):
                print("\n⏸️  Ready for next batch...")
                time.sleep(1)
        
        print("\n✅ All batches reviewed!")
        self.save_progress()

    def print_summary(self):
        """Print session summary"""
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total URLs: {self.stats['total']}")
        print(f"✅ Renamed (ready): {self.stats['renamed']}")
        print(f"📥 Downloaded (need renaming): {self.stats['downloaded']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"⏳ Pending: {self.stats['pending']}")
        
        if self.stats['renamed'] > 0:
            print(f"\n✅ {self.stats['renamed']} files are ready in: /app/output/manual_renamed/")
        
        if self.stats['pending'] > 0:
            print(f"\n⏳ {self.stats['pending']} URLs still need processing")
            print("   Run this script again to continue")

def main():
    print("🚀 Starting Manual Download Helper...")
    
    helper = ManualDownloadHelper()
    
    if not helper.downloads:
        print("❌ No downloads to process. Exiting.")
        return
    
    helper.interactive_download_session()
    helper.print_summary()

if __name__ == "__main__":
    main()

print("✅ Manual download helper loaded successfully")
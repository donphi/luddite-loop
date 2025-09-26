#!/usr/bin/env python3
# ============================================================================
# FILE: process_showcase_meta.py
# LOCATION: pipelines/01-data_ingestion/02-restructure_ukb_showcase_data/scripts/process_showcase_meta.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Restructures UK Biobank showcase data using Meta-Inventory acronyms with SPECTER 2 embeddings for optimal selection
# ============================================================================

"""
MODULE OVERVIEW:
This module processes UK Biobank showcase data using the Medical Abbreviation and Acronym Meta-Inventory:
1. Loads field, category, and encoding data from tab-separated files
2. Uses SPECTER 2 embeddings to select optimal acronyms from Meta-Inventory
3. Joins fields with categories and encoding metadata
4. Builds comprehensive field-to-dictionary codebook
5. Generates validation reports and human verification samples

CLASSES:
- AcronymEmbeddingSelector: Uses SPECTER 2 embeddings to select best acronyms by semantic similarity

METHODS:
- embed_batch(): Generates embeddings for text batches using SPECTER 2
- select_best_acronym(): Selects best acronym using embedding similarity
- process_phrase_group(): Processes acronym candidates for a phrase
- infer_column_dtype(): Infers pandas dtypes from sample data
- load_and_type_dataframe(): Loads TSV files with automatic type inference
- reorder_columns(): Reorders DataFrame columns
- move_after(): Moves columns to position after anchor column
- _norm(): Normalizes text for matching
- validate_acronym_match(): Validates if acronym is reasonable for phrase
- load_meta_inventory(): Loads Medical Abbreviation and Acronym Meta-Inventory
- build_comprehensive_meta_inventory_data(): Builds acronym mappings with embedding selection
- apply_acronyms_with_tracking(): Applies acronyms with comprehensive tracking
- main(): Main processing function

ROUTES:
- N/A (This is a data processing module, not a web service)

HYPERPARAMETERS:
- USE_COMPREHENSIVE_TRACKING: True (enables detailed acronym tracking)
- USE_EMBEDDINGS: True (enables SPECTER 2 embedding-based selection)
- META_INVENTORY_CFG: Dictionary of Meta-Inventory processing parameters
- MAX_ACRONYMS_PER_TITLE: 4 (maximum acronyms to replace per title)
- MIN_PHRASE_LEN: 2 (minimum phrase length to consider)
- MIN_ACRONYM_LEN: 2 (minimum acronym length)
- MAX_ACRONYM_LEN: 10 (maximum acronym length)
- EMBEDDING_MODEL: 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
- EMBEDDING_BATCH_SIZE: 128
- SIMILARITY_THRESHOLD: 0.65 (minimum similarity to accept)
- HIGH_CONFIDENCE_THRESHOLD: 0.91 (auto-accept threshold)

SEEDS:
- RANDOM_SEED: Not explicitly set (uses default numpy random state)
- NP_SEED: Not explicitly set

DEPENDENCIES:
- pandas==2.1.4
- pyarrow==14.0.2
- numpy==1.26.2
- pyahocorasick>=2.1.0 (optional, for substring matching)
- requests>=2.31.0
- python-dotenv>=1.0.0
- transformers>=4.35.0
- sentence-transformers>=2.2.2
- tokenizers>=0.15.0
- torch>=2.0.0 (for GPU acceleration)
- scikit-learn>=1.3.0 (for cosine similarity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import pickle
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from itertools import combinations

try:
    import ahocorasick  # pip install pyahocorasick
except Exception:
    ahocorasick = None

# ========================== CONFIGURATION ==========================
USE_COMPREHENSIVE_TRACKING = True
USE_EMBEDDINGS = True  # Toggle for embedding-based selection

# Meta-Inventory configuration
META_INVENTORY_CFG = {
    # Matching parameters
    "MAX_ACRONYMS_PER_TITLE": [1,2,3,4],  # Maximum acronyms to replace per title
                                  # Can be an integer (original behavior with single acr_abb column)
                                  # Or list/tuple like [1,2,3,4] to generate all combinations
                                  # e.g., 4 = single column with up to 4 replacements
                                  # [1,2,3,4] = multiple columns with all combinations up to 4
    "MIN_PHRASE_LEN": 2,          # Minimum phrase length to consider
    "MIN_ACRONYM_LEN": 2,          # Minimum acronym length
    "MAX_ACRONYM_LEN": 10,         # Maximum acronym length
    "SKIP_EMBEDDING_FOR_SINGLE_LETTERS": True,  # Single letters have poor embedding similarity

    # Normalization parameters
    "NORMALIZE_CONNECTORS": True,  # Normalize prepositions and articles for better matching
    "CONNECTOR_WORDS": ['at', 'in', 'on', 'of', 'by', 'to', 'for', 'with', 'from', 'as', 'is', 'the', 'a', 'an'],
    "NORMALIZE_NUMBERS": True,  # Convert "one" -> "1", "two" -> "2", etc.
    
    # Source priorities (higher = better) - used as fallback when embeddings unavailable
    "SOURCE_SCORES": {
        'UMLS': 100,
        'ADAM': 90,
        'Berman': 85,
        'Vanderbilt': 80,
        'Vanderbilt Clinic Notes': 80,
        'Vanderbilt Discharge Sums': 80,
        'Columbia': 70,
        'Stetson': 60,
        'Wikipedia': 5,  # Low priority due to noise
    },
    
    # Filtering rules
    "BLOCK_SOURCES": [],  # Add sources to completely block, e.g., ['Wikipedia']
    "REQUIRE_WORD_BOUNDARY": True,  # Only match at word boundaries
    "MIN_PHRASE_WORDS": 1,  # Minimum number of words in phrase for replacement
    
    # Validation rules
    "MIN_LENGTH_RATIO": 0.3,  # Acronym must be < 30% of original phrase length
    "REQUIRE_CAPITAL_LETTERS": True,  # Acronym must have capital letters
    
    # Embedding configuration
    "EMBEDDING_MODEL": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",  # Can also use specter2_proximity, specter2_adhoc
    "EMBEDDING_BATCH_SIZE": 128,
    "SIMILARITY_THRESHOLD": 0.65,  # Minimum similarity to accept
    "HIGH_CONFIDENCE_THRESHOLD": 0.91,  # Auto-accept threshold
    "CACHE_EMBEDDINGS": True,
    "EMBEDDING_CACHE_DIR": Path("/output/embedding_cache"),
}

# ----------------------------- Embedding-based Acronym Selector -----------------------------

class AcronymEmbeddingSelector:
    """
    Uses SPECTER 2 embeddings to select the best acronym from multiple candidates
    by comparing semantic similarity in the scientific literature embedding space.
    """
    
    def __init__(self, model_name: str = None, cache_dir: Path = None, device: str = None):
        """
        Initialize the embedding selector with SPECTER 2 model.
        
        Args:
            model_name: HuggingFace model name (default: from config)
            cache_dir: Directory to cache embeddings
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        model_name = model_name or META_INVENTORY_CFG["EMBEDDING_MODEL"]
        self.cache_dir = cache_dir or META_INVENTORY_CFG["EMBEDDING_CACHE_DIR"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"🧠 Initializing SPECTER 2 embeddings on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize embedding cache
        self.embedding_cache = {}
        self.load_cache()
        
        print(f"  ✓ Model loaded: {model_name}")
        print(f"  ✓ Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_name(0)}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def load_cache(self):
        """Load cached embeddings from disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"  ✓ Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"  ⚠️ Could not load cache: {e}")
                self.embedding_cache = {}
    
    def save_cache(self):
        """Save embeddings cache to disk"""
        if not META_INVENTORY_CFG["CACHE_EMBEDDINGS"]:
            return
        
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"  ✓ Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"  ⚠️ Could not save cache: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        # Check cache first
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                embeddings.append((i, self.embedding_cache[cache_key]))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            batch_size = META_INVENTORY_CFG["EMBEDDING_BATCH_SIZE"]
            new_embeddings = []
            
            for i in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[i:i + batch_size]
                
                # Tokenize and encode
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use pooler_output if available, otherwise mean pooling
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        batch_embeddings = outputs.pooler_output
                    else:
                        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    new_embeddings.append(batch_embeddings)
            
            # Concatenate all new embeddings
            if new_embeddings:
                new_embeddings = np.vstack(new_embeddings)
                
                # Cache the new embeddings
                for text, embedding, idx in zip(texts_to_encode, new_embeddings, text_indices):
                    cache_key = self._get_cache_key(text)
                    self.embedding_cache[cache_key] = embedding
                    embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in embeddings])
    
    def select_best_acronym(
        self, 
        long_form: str, 
        acronym_candidates: List[Tuple[str, float]]
    ) -> Tuple[Optional[str], float, Dict]:
        """
        Select the best acronym for a given long form using embedding similarity.
        
        Args:
            long_form: The full phrase (e.g., "vitamin a")
            acronym_candidates: List of (acronym, source_score) tuples
            
        Returns:
            Tuple of (best_acronym, similarity_score, debug_info)
        """
        if not acronym_candidates:
            return None, 0.0, {}
        
        # Extract acronyms and scores
        acronyms = [acr for acr, _ in acronym_candidates]
        source_scores = [score for _, score in acronym_candidates]
        
        # Normalize the long form for better semantic matching
        normalized_long_form = _norm_semantic(long_form) if META_INVENTORY_CFG.get('NORMALIZE_CONNECTORS', True) else long_form
        
        # Generate embeddings for normalized long form and all candidates
        texts_to_embed = [normalized_long_form] + acronyms
        embeddings = self.embed_batch(texts_to_embed)
        
        long_embedding = embeddings[0:1]  # Keep as 2D array
        candidate_embeddings = embeddings[1:]
        
        # Calculate cosine similarities
        similarities = cosine_similarity(long_embedding, candidate_embeddings)[0]
        
        # Combine similarity with source scores (weighted)
        similarity_weight = 0.8  # Prioritize embedding similarity
        source_weight = 0.2
        
        # Normalize source scores to [0, 1]
        max_source = max(source_scores) if source_scores else 1.0
        if max_source == 0:
            # All scores are 0 (no recognized sources), treat them equally
            norm_source_scores = [0.5] * len(source_scores)  # Give all a neutral score
        else:
            norm_source_scores = [s / max_source for s in source_scores]
        
        # Calculate combined scores
        combined_scores = [
            similarity_weight * sim + source_weight * src
            for sim, src in zip(similarities, norm_source_scores)
        ]
        
        # Find best candidate
        best_idx = np.argmax(combined_scores)
        best_similarity = similarities[best_idx]
        
        # Apply threshold
        # Special handling for critical medical laterality terms
        if long_form in ['left', 'right'] and acronyms[best_idx] in ['L', 'R']:
            print(f"  🏥 Medical laterality override: '{long_form}' → '{acronyms[best_idx]}' (similarity: {best_similarity:.3f})")
            return acronyms[best_idx], best_similarity, {
                "all_similarities": dict(zip(acronyms, similarities)),
                "override": "medical_laterality",
                "confidence": "high"
            }
        
        # Apply threshold
        if best_similarity < META_INVENTORY_CFG["SIMILARITY_THRESHOLD"]:
            return None, best_similarity, {
                "all_similarities": dict(zip(acronyms, similarities)),
                "rejected_reason": f"Below threshold ({best_similarity:.3f} < {META_INVENTORY_CFG['SIMILARITY_THRESHOLD']})"
            }
        
        debug_info = {
            "all_similarities": dict(zip(acronyms, similarities)),
            "combined_scores": dict(zip(acronyms, combined_scores)),
            "selected": acronyms[best_idx],
            "confidence": "high" if best_similarity > META_INVENTORY_CFG["HIGH_CONFIDENCE_THRESHOLD"] else "medium"
        }
        
        return acronyms[best_idx], best_similarity, debug_info
    
    def process_phrase_group(
        self,
        norm_phrase: str,
        candidates: List[Tuple]
    ) -> Tuple[str, float]:
        """
        Process a group of acronym candidates for a normalized phrase.
        
        Args:
            norm_phrase: Normalized long form phrase
            candidates: List of candidate tuples (acronym, cui, source, tty, sty_list, score)
            
        Returns:
            Tuple of (best_acronym, similarity_score)
        """
        # Extract acronyms and scores for embedding selection
        acronym_score_pairs = [(c[0], c[5]) for c in candidates]
        
        # Use embedding selector to find best
        best_acr, similarity, debug = self.select_best_acronym(norm_phrase, acronym_score_pairs)
        
        if best_acr:
            # Print debug info for important medical terms
            if any(term in norm_phrase for term in ['vitamin', 'blood', 'pressure', 'white blood cell']):
                print(f"\n  📊 '{norm_phrase}' → '{best_acr}' (similarity: {similarity:.3f})")
                for acr, sim in debug['all_similarities'].items():
                    mark = "✓" if acr == best_acr else " "
                    print(f"    {mark} {acr}: {sim:.3f}")
        
        return best_acr, similarity

# ----------------------------- Helper Functions -----------------------------

def infer_column_dtype(series, col_name):
    if series.isna().all():
        return 'object'
    sample = series.dropna()
    if len(sample) == 0:
        return 'object'
    if 'date' in col_name.lower() or 'debut' in col_name or 'version' in col_name:
        try:
            pd.to_datetime(sample.head(100))
            return 'datetime64[ns]'
        except:
            pass
    try:
        numeric_sample = pd.to_numeric(sample, errors='coerce')
        if not numeric_sample.isna().any():
            if (numeric_sample % 1 == 0).all():
                return 'int64'
            else:
                return 'float64'
    except:
        pass
    return 'object'

def load_and_type_dataframe(filepath, delimiter='\t'):
    print(f"\n📂 Loading {filepath.name}...")
    df_str = pd.read_csv(filepath, delimiter=delimiter, dtype=str, na_values=['', 'NA', 'null', 'None'])
    print(f"  Shape: {df_str.shape}")
    print(f"  Columns: {', '.join(df_str.columns[:5])}...")
    dtype_dict = {}
    for col in df_str.columns:
        inferred_type = infer_column_dtype(df_str[col], col)
        dtype_dict[col] = inferred_type
        print(f"  ✓ {col}: {inferred_type}")
    df_typed = pd.DataFrame()
    for col in df_str.columns:
        if dtype_dict[col] == 'datetime64[ns]':
            df_typed[col] = pd.to_datetime(df_str[col], errors='coerce')
        elif dtype_dict[col] in ['int64', 'float64']:
            df_typed[col] = pd.to_numeric(df_str[col], errors='coerce')
            if dtype_dict[col] == 'int64':
                df_typed[col] = df_typed[col].astype('Int64')
        else:
            df_typed[col] = df_str[col]
    return df_typed, dtype_dict

def reorder_columns(df, first_cols):
    all_cols = df.columns.tolist()
    remaining_cols = [col for col in all_cols if col not in first_cols]
    existing_first_cols = [col for col in first_cols if col in all_cols]
    return df[existing_first_cols + remaining_cols]

def move_after(df: pd.DataFrame, anchor_col: str, cols_to_move: list) -> pd.DataFrame:
    cols_to_move = [c for c in cols_to_move if c in df.columns and c != anchor_col]
    if not cols_to_move:
        return df
    base_cols = [c for c in df.columns if c not in cols_to_move]
    try:
        idx = base_cols.index(anchor_col)
    except ValueError:
        return df[base_cols + cols_to_move]
    new_order = base_cols[:idx+1] + cols_to_move + base_cols[idx+1:]
    return df[new_order]

# ----------------------------- Meta-Inventory functions -----------------------------

def _norm(s: str):
    """Normalize string for matching"""
    if pd.isna(s) or s is None:
        return None
    s = str(s).lower().strip()
    s = s.replace('-', ' ').replace('/', ' ')
    return ' '.join(s.split())

def _norm_semantic(s: str) -> str:
    """Normalize string for semantic comparison by removing connector words and normalizing numbers"""
    if pd.isna(s) or s is None:
        return None
    
    s = _norm(s)  # Apply basic normalization first
    
    cfg = META_INVENTORY_CFG
    
    if cfg.get('NORMALIZE_NUMBERS', True):
        # Convert written numbers to digits
        number_map = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th'
        }
        for word, digit in number_map.items():
            s = re.sub(r'\b' + word + r'\b', digit, s, flags=re.IGNORECASE)
    
    if cfg.get('NORMALIZE_CONNECTORS', True):
        # Remove connector words but preserve word boundaries
        connector_words = cfg.get('CONNECTOR_WORDS', [])
        
        # Split into words
        words = s.split()
        
        # Filter out connector words while preserving important content
        filtered_words = []
        for i, word in enumerate(words):
            # Keep the word if it's not a connector OR if it's critical for meaning
            # (e.g., keep numbers and important medical terms)
            if word not in connector_words or word.isdigit() or len(word) > 3:
                filtered_words.append(word)
        
        # Rejoin, preserving at least minimal structure
        s = ' '.join(filtered_words)
    
    # Clean up any extra spaces
    s = ' '.join(s.split())
    
    return s

def validate_acronym_match(acronym: str, phrase: str) -> bool:
    """Validate if an acronym is reasonable for a given phrase"""
    cfg = META_INVENTORY_CFG
    
    # Special exception for medical laterality
    if phrase.lower() in ['left', 'right'] and acronym in ['L', 'R']:
        return True
    
    phrase_clean = phrase.replace(' ', '').replace('-', '').replace('/', '')
    if len(acronym) >= len(phrase_clean) * cfg['MIN_LENGTH_RATIO']:
        return False
    
    phrase_words = phrase.split()
    if len(phrase_words) < cfg['MIN_PHRASE_WORDS']:
        return False
    
    if cfg['REQUIRE_CAPITAL_LETTERS'] and not any(c.isupper() for c in acronym):
        return False
    
    if len(acronym) > len(phrase_words) + 2:
        return False
    
    return True

def load_meta_inventory(meta_inv_file: Path) -> pd.DataFrame:
    """Load Medical Abbreviation and Acronym Meta-Inventory"""
    print(f"\n🔤 Loading Meta-Inventory from {meta_inv_file.name}...")
    
    try:
        meta_df = pd.read_csv(meta_inv_file, sep='|', dtype=str, on_bad_lines='skip')
    except:
        try:
            meta_df = pd.read_csv(meta_inv_file, sep='\t', dtype=str, on_bad_lines='skip')
        except:
            try:
                meta_df = pd.read_csv(meta_inv_file, dtype=str, on_bad_lines='skip')
            except Exception as e:
                print(f"❌ ERROR loading Meta-Inventory: {e}")
                return pd.DataFrame()
    
    print(f"  Shape: {meta_df.shape}")
    print(f"  Columns: {list(meta_df.columns)}")
    
    required = ['SF', 'LF', 'NormLF']
    missing = [c for c in required if c not in meta_df.columns]
    if missing:
        print(f"❌ ERROR: Missing required columns: {missing}")
        return pd.DataFrame()
    
    meta_df = meta_df.dropna(subset=['SF', 'LF'])
    meta_df['SF'] = meta_df['SF'].str.strip()
    meta_df['LF'] = meta_df['LF'].str.strip()
    meta_df['NormLF'] = meta_df['LF'].apply(_norm)

    if META_INVENTORY_CFG.get('NORMALIZE_CONNECTORS', True):
        meta_df['NormLF_Semantic'] = meta_df['NormLF'].apply(_norm_semantic)
    else:
        meta_df['NormLF_Semantic'] = meta_df['NormLF']
    
    if 'NormSF' not in meta_df.columns:
        meta_df['NormSF'] = meta_df['SF'].str.upper()
    
    cfg = META_INVENTORY_CFG
    if cfg['BLOCK_SOURCES'] and 'Source' in meta_df.columns:
        meta_df = meta_df[~meta_df['Source'].isin(cfg['BLOCK_SOURCES'])]
    
    # Special exception for medical laterality terms
    laterality_mask = (
        (meta_df['NormLF'].isin(['left', 'right'])) & 
        (meta_df['SF'].isin(['L', 'R']))
    )
    
    meta_df = meta_df[
        laterality_mask |  # Allow L/R for left/right regardless of MIN_ACRONYM_LEN
        (
            (meta_df['SF'].str.len() >= cfg['MIN_ACRONYM_LEN']) &
            (meta_df['SF'].str.len() <= cfg['MAX_ACRONYM_LEN']) &
            (meta_df['NormLF'].str.len() >= cfg['MIN_PHRASE_LEN'])
        )
    ]
    print(f"\n🔍 Single letters in meta: L={any(meta_df['SF']=='L')}, R={any(meta_df['SF']=='R')}")

    if 'Source' in meta_df.columns:
        meta_df['score'] = meta_df['Source'].map(cfg['SOURCE_SCORES']).fillna(0)
    else:
        meta_df['score'] = 50
    
    valid_mask = meta_df.apply(
        lambda row: validate_acronym_match(row['SF'], row['NormLF']), 
        axis=1
    )
    meta_df = meta_df[valid_mask]
    
    print(f"  ✓ After filtering: {len(meta_df):,} valid mappings")
    
    return meta_df

def build_comprehensive_meta_inventory_data(meta_df: pd.DataFrame, embedding_selector=None) -> tuple:
    """Build comprehensive mapping with tracking data and optional embedding selection"""
    if meta_df.empty:
        return {}, {}
    
    cfg = META_INVENTORY_CFG

    # Group by semantic normalized long form for better matching
    group_col = 'NormLF_Semantic' if 'NormLF_Semantic' in meta_df.columns else 'NormLF'
    grouped = meta_df.groupby(group_col)
    
    full2candidates = {}
    full2best = {}
    
    # Process each phrase group
    print("\n🔄 Processing phrase groups...")
    total_groups = len(grouped)
    
    for i, (norm_phrase_semantic, group) in enumerate(grouped):
        # Get all the original normalized forms that map to this semantic form
        original_norm_phrases = group['NormLF'].unique()
        
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{total_groups} phrase groups...")
        
        if pd.isna(norm_phrase_semantic) or not norm_phrase_semantic:
            continue
        
        if cfg['MIN_PHRASE_WORDS'] > 1 and ' ' not in norm_phrase_semantic:
            continue
        
        # Create candidate list
        candidates = []
        seen_acronyms = set()
        
        for _, row in group.iterrows():
            acronym = row['SF'].upper()
            
            if acronym in seen_acronyms:
                continue
            seen_acronyms.add(acronym)
            
            source = row.get('Source', 'Unknown')
            score = row.get('score', 50)
            
            if not validate_acronym_match(acronym, original_norm_phrases[0]):  # Validate against first original form
                continue
            
            candidates.append((
                acronym,           # acronym
                '',               # cui (not used)
                source,           # source
                'META',           # type marker
                [],               # sty_list (not used)
                score             # score
            ))
        
        if not candidates:
            continue

        if norm_phrase_semantic in ['left', 'right']:
            print(f"🔍 {norm_phrase_semantic}: candidates={[(c[0], c[5]) for c in candidates]}")
        
        embedding_selected = False
        # Use embedding selector if available and enabled
        if USE_EMBEDDINGS and embedding_selector and len(candidates) > 1:
            # Use the semantic normalized form for embedding comparison
            best_acr, similarity = embedding_selector.process_phrase_group(norm_phrase_semantic, candidates)
            if best_acr and similarity >= META_INVENTORY_CFG["SIMILARITY_THRESHOLD"]:
                # Reorder candidates to put embedding-selected best first
                candidates = sorted(candidates, 
                                key=lambda x: (x[0] != best_acr, -x[5], len(x[0]), x[0]))
                embedding_selected = True
        
        # Always sort by score if embeddings didn't select a winner
        if not embedding_selected:
            candidates.sort(key=lambda x: (-x[5], len(x[0]), x[0]))
        
        # Keep only top N candidates
        candidates = candidates[:10]
        
        # Store mappings for all original normalized forms that map to this semantic form
        for original_norm in original_norm_phrases:
            full2candidates[original_norm] = candidates
            if candidates:
                full2best[original_norm] = candidates[0][0]
    
    print(f"  ✓ Created {len(full2best):,} unique phrase→acronym mappings")
    
    # Save embedding cache if used
    if embedding_selector:
        embedding_selector.save_cache()
    
    # Debug sample mappings
    print("\n🔍 Sample mappings:")
    samples = list(full2best.items())[:20]
    for phrase, acr in samples:
        print(f"  '{phrase}' → '{acr}'")
    
    return full2candidates, full2best


def apply_acronyms_with_tracking(fields_df: pd.DataFrame, full2candidates: dict) -> tuple:
    """Apply acronyms with comprehensive tracking and limit per title
    
    If MAX_ACRONYMS_PER_TITLE is a list/tuple, generates all combinations.
    If it's an integer, uses original single-column behavior.
    """
    if not ahocorasick:
        print("⚠️ pyahocorasick required for substring matching!")
        if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
            max_combos = sum(range(1, max(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE']) + 1))
            empty_cols = {f'acr_abb_{i+1}': pd.Series([""] * len(fields_df), dtype='string') 
                         for i in range(max_combos)}
            return empty_cols, pd.DataFrame()
        else:
            return pd.Series([""] * len(fields_df), dtype='string'), pd.DataFrame()
    
    cfg = META_INVENTORY_CFG
    
    # Determine if we're in multi-column mode
    multi_column_mode = isinstance(cfg['MAX_ACRONYMS_PER_TITLE'], (list, tuple))
    if multi_column_mode:
        max_acronyms = max(cfg['MAX_ACRONYMS_PER_TITLE'])
        # Calculate total columns needed (sum of 1 to max_acronyms)
        total_columns = 2**max_acronyms - 1
    else:
        max_acronyms = cfg['MAX_ACRONYMS_PER_TITLE']
        total_columns = 1
    
    # Build the automaton
    A = ahocorasick.Automaton()
    for full_phrase, candidates in full2candidates.items():
        if full_phrase and len(full_phrase) >= cfg['MIN_PHRASE_LEN']:
            A.add_word(full_phrase, (full_phrase, candidates))
    A.make_automaton()
    
    # Initialize result storage
    if multi_column_mode:
        all_results = {f'acr_abb_{i+1}': [] for i in range(total_columns)}
    else:
        applied = []
    
    all_candidates_list = []
    
    # Global mapping for consistency
    global_phrase_to_acronym = {}
    for phrase, candidates in full2candidates.items():
        if candidates:
            global_phrase_to_acronym[phrase] = candidates[0][0]
    
    for idx, row in fields_df.iterrows():
        title = row['title']
        field_id = row['field_id']
        
        if pd.isna(title):
            if multi_column_mode:
                for col in all_results:
                    all_results[col].append("")
            else:
                applied.append("")
            continue
        
        norm_title = _norm(str(title))
        if not norm_title:
            if multi_column_mode:
                for col in all_results:
                    all_results[col].append("")
            else:
                applied.append("")
            continue
        
        # Find ALL matches in this title
        matches = []
        for end_pos, payload in A.iter(norm_title):
            matched_phrase, candidates = payload
            start_pos = end_pos - len(matched_phrase) + 1
            
            # Check word boundaries if required
            if cfg['REQUIRE_WORD_BOUNDARY']:
                if start_pos > 0 and norm_title[start_pos-1].isalnum():
                    continue
                if end_pos < len(norm_title) - 1 and norm_title[end_pos+1].isalnum():
                    continue
            
            matches.append((start_pos, end_pos, matched_phrase, candidates))
        
        if not matches:
            if multi_column_mode:
                for col in all_results:
                    all_results[col].append("")
            else:
                applied.append("")
            continue
        
        # Sort by length (longest first) then by position
        matches.sort(key=lambda x: (-(x[1] - x[0] + 1), x[0]))
        
        # Select non-overlapping matches, preferring longer ones
        chosen_matches = []
        taken = [False] * len(norm_title)
        
        for start, end, phrase, candidates in matches:
            if any(taken[i] for i in range(start, end + 1)):
                continue
            
            for i in range(start, end + 1):
                taken[i] = True
            
            chosen_matches.append((start, end, phrase, candidates))
            
            if len(chosen_matches) >= max_acronyms:
                break
        
        if not chosen_matches:
            if multi_column_mode:
                for col in all_results:
                    all_results[col].append("")
            else:
                applied.append("")
            continue
        
        # Sort by position for correct replacement
        chosen_matches.sort(key=lambda x: x[0])
        
        # Build replacement data
        replacements = []
        for start, end, phrase, candidates in chosen_matches:
            best_acronym = global_phrase_to_acronym.get(phrase)
            if best_acronym:
                tokens = phrase.split()
                pattern = r'\b' + r'[\s\-/]*'.join(re.escape(tok) for tok in tokens) + r'\b'
                replacements.append((pattern, best_acronym, phrase, candidates))
        
        if multi_column_mode:
            # Generate all combinations
            column_idx = 0
            for combo_size in range(1, min(len(replacements) + 1, max_acronyms + 1)):
                for combo in combinations(range(len(replacements)), combo_size):
                    if column_idx >= total_columns:  # Stop if we've filled all columns
                        break
                    # Apply this combination
                    replaced_title = title
                    for i in combo:
                        pattern_str, acronym, phrase, candidates = replacements[i]
                        pattern = re.compile(pattern_str, re.IGNORECASE)
                        replaced_title = pattern.sub(acronym, replaced_title)
                        
                        # Track candidates for the first occurrence only
                        if column_idx == 0:
                            for rank, cand in enumerate(candidates[:5], 1):
                                if len(cand) >= 6:
                                    acr, cui, sab, tty, sty_list, score = cand
                                    all_candidates_list.append({
                                        'field_id': field_id,
                                        'title': title,
                                        'matched_phrase': phrase,
                                        'replaced_title': replaced_title,
                                        'candidate_acronym': acr,
                                        'cui': cui,
                                        'sab': sab,
                                        'tty': tty,
                                        'sty': '|'.join(sty_list) if sty_list else "",
                                        'score': score,
                                        'rank': rank,
                                        'chosen': rank == 1,
                                        'has_conflict': False,
                                        'domain': 'medical'
                                    })
                    
                    # Remove redundant patterns
                    redundant_pattern = re.compile(r'\b(\w+)\s*\(\s*\1\s*\)', re.IGNORECASE)
                    replaced_title = redundant_pattern.sub(r'\1', replaced_title)
                    
                    all_results[f'acr_abb_{column_idx + 1}'].append(
                        replaced_title if replaced_title != title else ""
                    )
                    column_idx += 1
            
            # Fill remaining columns with empty strings
            while column_idx < total_columns:
                all_results[f'acr_abb_{column_idx + 1}'].append("")
                column_idx += 1
        
        else:
            # Original single-column behavior
            replaced_title = title
            for pattern_str, acronym, phrase, candidates in replacements:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                replaced_title = pattern.sub(acronym, replaced_title)
                
                # Track all candidates for reporting
                for rank, cand in enumerate(candidates[:5], 1):
                    if len(cand) >= 6:
                        acr, cui, sab, tty, sty_list, score = cand
                        all_candidates_list.append({
                            'field_id': field_id,
                            'title': title,
                            'matched_phrase': phrase,
                            'replaced_title': replaced_title,
                            'candidate_acronym': acr,
                            'cui': cui,
                            'sab': sab,
                            'tty': tty,
                            'sty': '|'.join(sty_list) if sty_list else "",
                            'score': score,
                            'rank': rank,
                            'chosen': rank == 1,
                            'has_conflict': False,
                            'domain': 'medical'
                        })
            
            # Remove redundant patterns
            redundant_pattern = re.compile(r'\b(\w+)\s*\(\s*\1\s*\)', re.IGNORECASE)
            replaced_title = redundant_pattern.sub(r'\1', replaced_title)
            
            if replaced_title != title:
                applied.append(replaced_title)
            else:
                applied.append("")
    
    candidates_df = pd.DataFrame(all_candidates_list)
    
    if multi_column_mode:
        # Convert lists to Series
        result_series = {col: pd.Series(data, dtype='string') 
                        for col, data in all_results.items()}
        return result_series, candidates_df
    else:
        return pd.Series(applied, dtype='string'), candidates_df

# ----------------------------- dictionary helpers (unchanged) -----------------------------

ENC_SYSTEM_MAP = {
    19: "ICD10",
    87: "ICD9",
    240: "OPCS4",
    6: "NonCancerIllness",
    5: "Operation",
    4: "Treatment",
    3: "Cancer",
    2: "Employment",
    1: "YesTruePresence",
    7: "YesNo",
    8: "CalendarMonth",
}

def is_hierarchical(structure_value: str) -> bool:
    try:
        return int(structure_value) == 2
    except Exception:
        return False

def normalize_hier_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        'value': 'coding',
        'meaning': 'meaning',
        'code_id': 'node_id',
        'parent_id': 'parent_id',
        'selectable': 'selectable'
    }).copy()
    out['coding'] = out['coding'].astype(str)
    if out['selectable'].dtype == 'object':
        out['selectable'] = out['selectable'].map(lambda x: 1 if str(x).strip().upper() in ('1','Y','YES','TRUE') else 0)
    else:
        out['selectable'] = pd.to_numeric(out['selectable'], errors='coerce').fillna(0).astype(int)
    out['encoding_id'] = pd.to_numeric(out['encoding_id'], errors='coerce').astype('Int64')
    out['node_id']     = pd.to_numeric(out['node_id'], errors='coerce').astype('Int64')
    out['parent_id']   = pd.to_numeric(out['parent_id'], errors='coerce').astype('Int64')
    return out[['encoding_id','coding','meaning','node_id','parent_id','selectable']]

def normalize_flat_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={'value':'coding','meaning':'meaning'}).copy()
    out['encoding_id'] = pd.to_numeric(out['encoding_id'], errors='coerce').astype('Int64')
    out['coding']   = out['coding'].astype(str)
    out['meaning']  = out['meaning'].astype(str)
    out['node_id']  = pd.Series([pd.NA]*len(out), dtype='Int64')
    out['parent_id']= pd.Series([pd.NA]*len(out), dtype='Int64')
    out['selectable']= 1
    return out[['encoding_id','coding','meaning','node_id','parent_id','selectable']]

# ----------------------------- main -----------------------------

def main():
    global USE_EMBEDDINGS
    print("🔬 UK Biobank Field / Category / Dictionary Linker")
    print("   Using Medical Abbreviation and Acronym Meta-Inventory")
    if USE_EMBEDDINGS:
        print("   Enhanced with SPECTER 2 Embeddings")
    print("=" * 50)

    input_dir = Path('/input')
    output_dir = Path('/output')
    output_dir.mkdir(exist_ok=True)

    # core files
    field_file = input_dir / 'field.txt'
    category_file = input_dir / 'category.txt'
    encoding_file = input_dir / 'encoding.txt'
    
    # dictionary files
    enhierstring_file = input_dir / 'enhierstring.txt'
    enhierint_file = input_dir / 'enhierint.txt'
    esimpint_file = input_dir / 'esimpint.txt'
    esimpstring_file = input_dir / 'esimpstring.txt'
    esimpreal_file = input_dir / 'esimpreal.txt'
    esimpdate_file = input_dir / 'esimpdate.txt'
    esimptime_file = input_dir / 'esimptime.txt'
    
    # Meta-Inventory file
    meta_inv_file = None
    possible_names = [
        'Metainventory_Version1.0.0.csv',
        'Metainventory_Version1.0.1.csv',
        'Metainventory_Version1.0.2.csv',
        'metainventory.csv',
        'meta-inventory.csv',
        'meta_inventory.csv',
        'metainventory.txt',
        'meta-inventory.txt',
        'meta_inventory.txt',
    ]
    
    for name in possible_names:
        test_file = input_dir / name
        if test_file.exists():
            meta_inv_file = test_file
            print(f"  Found Meta-Inventory: {name}")
            break
    
    if not meta_inv_file:
        print("\n⚠️ Meta-Inventory file not found!")
        print(f"  Expected in: {input_dir}")
        print(f"  Tried names: {', '.join(possible_names)}")

    # Validate core files
    missing = [p.name for p in [field_file, category_file, encoding_file] if not p.exists()]
    if missing:
        print(f"❌ ERROR: missing required file(s): {missing}")
        sys.exit(1)

    value_files = [f for f in [enhierstring_file, enhierint_file, esimpint_file, 
                               esimpstring_file, esimpreal_file, esimpdate_file, 
                               esimptime_file] if f.exists()]
    if not value_files:
        print("❌ ERROR: No dictionary value files found.")
        sys.exit(1)

    # Load core data
    fields_df, field_types = load_and_type_dataframe(field_file)
    categories_df, cat_types = load_and_type_dataframe(category_file)
    encoding_df, enc_types = load_and_type_dataframe(encoding_file)

    # Initialize embedding selector if enabled
    embedding_selector = None
    if USE_EMBEDDINGS:
        try:
            embedding_selector = AcronymEmbeddingSelector(cache_dir=output_dir / "embedding_cache")
        except Exception as e:
            print(f"⚠️ Could not initialize embeddings: {e}")
            print("  Falling back to score-based selection")
            USE_EMBEDDINGS = False

    # Load Meta-Inventory abbreviations
    if meta_inv_file and meta_inv_file.exists():
        meta_df = load_meta_inventory(meta_inv_file)
        
        if not meta_df.empty:
            if USE_COMPREHENSIVE_TRACKING:
                full2candidates, full2acr = build_comprehensive_meta_inventory_data(
                    meta_df, embedding_selector
                )
                acr_result, candidates_df = apply_acronyms_with_tracking(
                    fields_df, full2candidates
                )
                
                # Handle both single and multi-column modes
                if isinstance(acr_result, dict):
                    # Multi-column mode
                    for col_name, col_data in acr_result.items():
                        fields_df[col_name] = col_data
                else:
                    # Single-column mode
                    fields_df['acr_abb'] = acr_result
                
                if not candidates_df.empty:
                    candidates_df.to_parquet(output_dir / 'acronym_candidates.parquet')
                    candidates_df.to_csv(output_dir / 'acronym_candidates.csv', index=False)
                    print(f"  ✓ Saved {len(candidates_df)} acronym candidates")
            else:
                full2acr = {phrase: cands[0][0] for phrase, cands in 
                           build_comprehensive_meta_inventory_data(meta_df, embedding_selector)[0].items()}
                
                applied_acr = []
                for title in fields_df['title']:
                    if pd.isna(title):
                        applied_acr.append("")
                        continue
                    
                    norm_title = _norm(str(title))
                    result = title
                    replacements_made = 0
                    
                    for phrase, acronym in full2acr.items():
                        if replacements_made >= META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE']:
                            break
                        
                        if phrase in norm_title:
                            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                            result = pattern.sub(acronym, result)
                            replacements_made += 1
                    
                    applied_acr.append(result if result != title else "")
                
                fields_df['acr_abb'] = pd.Series(applied_acr, dtype='string')
        else:
            print("⚠️ Meta-Inventory loaded but empty after filtering")
            fields_df['acr_abb'] = pd.Series([""] * len(fields_df), dtype='string')
    else:
        print(f"\n⚠️ Meta-Inventory file not found, skipping abbreviation matching")
        fields_df['acr_abb'] = pd.Series([""] * len(fields_df), dtype='string')

    # Move acr_abb column after title
    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        max_val = max(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'])
        total_cols = 2**max_val - 1
        acr_cols = [f'acr_abb_{i+1}' for i in range(total_cols)]
        fields_df = move_after(fields_df, 'title', acr_cols)
    else:
        fields_df = move_after(fields_df, 'title', ['acr_abb'])

    # ========== REST OF PROCESSING (unchanged) ==========
    # [Keep all the rest of the code unchanged - category joining, dictionary processing, saving outputs, etc.]
    
    print("\n🔎 Analyzing category linkage...")
    fields_df['main_category_numeric'] = pd.to_numeric(fields_df['main_category'], errors='coerce')
    no_category_mask = fields_df['main_category_numeric'].isna()
    no_category_count = no_category_mask.sum()
    fields_df.loc[no_category_mask, 'main_category_numeric'] = -1
    print(f"  • Fields with NO main_category (null/empty): {no_category_count}")

    categories_df = categories_df.rename(columns={'title': 'category_title','notes': 'category_notes'})
    categories_df['category_id'] = pd.to_numeric(categories_df['category_id'], errors='coerce')
    valid_category_ids = set(categories_df['category_id'].dropna().unique())
    has_category_mask = ~no_category_mask
    unmatched_categories = fields_df[has_category_mask]['main_category_numeric'].unique()
    unmatched_categories = [c for c in unmatched_categories if c not in valid_category_ids]
    unmatched_count = fields_df[fields_df['main_category_numeric'].isin(unmatched_categories)].shape[0]
    print(f"  • Fields with main_category that doesn't exist in category.txt: {unmatched_count}")

    print("\n🔗 Joining field and category data...")
    merged_df = fields_df.merge(
        categories_df[['category_id', 'category_title', 'category_notes']],
        left_on='main_category_numeric',
        right_on='category_id',
        how='left'
    )

    no_cat_mask = (fields_df['main_category'].isna()) | (fields_df['main_category'] == '')
    merged_df.loc[no_cat_mask, 'category_title'] = 'No Category Assigned'
    merged_df.loc[no_cat_mask, 'category_notes'] = 'Field has no main_category value'
    unmatched_mask = (~no_cat_mask) & (merged_df['category_title'].isna())
    merged_df.loc[unmatched_mask, 'category_title'] = 'Unmatched Category'
    merged_df.loc[unmatched_mask, 'category_notes'] = 'Category ID exists in field but not found in category.txt'

    merged_df = merged_df.drop(['main_category_numeric'], axis=1)
    if 'category_id' in merged_df.columns:
        merged_df['category_id'] = merged_df['main_category']
        merged_df = merged_df.drop(['main_category'], axis=1)
    else:
        merged_df = merged_df.rename(columns={'main_category': 'category_id'})
    merged_df['category_id'] = pd.to_numeric(merged_df['category_id'], errors='coerce').fillna(-1).astype('int64')

    print("\n📚 Enriching fields with dictionary metadata...")
    enc_keep = encoding_df[['encoding_id','title','coded_as','structure','num_members','descript']].copy()
    enc_keep = enc_keep.rename(columns={
        'title': 'encoding_title',
        'coded_as': 'encoding_coded_as',
        'structure': 'encoding_structure',
        'num_members': 'encoding_num_members',
        'descript': 'encoding_description'
    })
    enc_keep['encoding_id'] = pd.to_numeric(enc_keep['encoding_id'], errors='coerce').astype('Int64')
    enc_keep['encoding_has_hierarchy'] = enc_keep['encoding_structure'].apply(lambda v: is_hierarchical(str(v)))
    enc_keep['encoding_system'] = enc_keep['encoding_id'].map(ENC_SYSTEM_MAP).fillna('Other')

    merged_df['encoding_id'] = pd.to_numeric(merged_df['encoding_id'], errors='coerce').astype('Int64')
    enriched = merged_df.merge(enc_keep, on='encoding_id', how='left')

    to_place = [
        'encoding_title','encoding_coded_as','encoding_structure','encoding_num_members',
        'encoding_has_hierarchy','encoding_system','encoding_description'
    ]
    enriched = move_after(enriched, 'encoding_id', to_place)

    # Column order
    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        max_val = max(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'])
        total_cols = 2**max_val - 1
        acr_cols = [f'acr_abb_{i+1}' for i in range(total_cols)]
        first_cols = ['field_id', 'title'] + acr_cols + ['category_id', 'category_title']
    else:
        first_cols = ['field_id', 'title', 'acr_abb', 'category_id', 'category_title']
    enriched = reorder_columns(enriched, first_cols)
    enriched = enriched.loc[:, ~enriched.columns.duplicated()]

    print("\n🧭 Building unified dictionary values…")
    dict_frames = []

    if enhierstring_file.exists():
        df, _ = load_and_type_dataframe(enhierstring_file)
        dict_frames.append(normalize_hier_df(df))
    if enhierint_file.exists():
        df, _ = load_and_type_dataframe(enhierint_file)
        dict_frames.append(normalize_hier_df(df))
    if esimpint_file.exists():
        df, _ = load_and_type_dataframe(esimpint_file)
        dict_frames.append(normalize_flat_df(df))
    if esimpstring_file.exists():
        df, _ = load_and_type_dataframe(esimpstring_file)
        dict_frames.append(normalize_flat_df(df))
    if esimpreal_file.exists():
        df, _ = load_and_type_dataframe(esimpreal_file)
        dict_frames.append(normalize_flat_df(df))
    if esimpdate_file.exists():
        df, _ = load_and_type_dataframe(esimpdate_file)
        dict_frames.append(normalize_flat_df(df))
    if esimptime_file.exists():
        df, _ = load_and_type_dataframe(esimptime_file)
        dict_frames.append(normalize_flat_df(df))

    dictionary_values = pd.concat(dict_frames, ignore_index=True).drop_duplicates()
    enc_used = set(enriched['encoding_id'].dropna().unique().tolist())
    dictionary_values = dictionary_values[dictionary_values['encoding_id'].isin(enc_used)]

    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        max_val = max(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'])
        total_cols = 2**max_val - 1
        acr_cols = [f'acr_abb_{i+1}' for i in range(total_cols)]
        # For codebook, you might want just the first column or all columns
        field_core = enriched[['field_id','title'] + acr_cols + ['encoding_id']].drop_duplicates()
    else:
        field_core = enriched[['field_id','title','acr_abb','encoding_id']].drop_duplicates()

    field_codebook = field_core.merge(dictionary_values, on='encoding_id', how='left')
    
    field_codebook = field_codebook.merge(
        enc_keep[['encoding_id','encoding_system']],
        on='encoding_id', how='left'
    )

    field_codebook['field_id']    = pd.to_numeric(field_codebook['field_id'], errors='coerce').astype('Int64')
    field_codebook['encoding_id'] = pd.to_numeric(field_codebook['encoding_id'], errors='coerce').astype('Int64')
    field_codebook['node_id']     = pd.to_numeric(field_codebook['node_id'], errors='coerce').astype('Int64')
    field_codebook['parent_id']   = pd.to_numeric(field_codebook['parent_id'], errors='coerce').astype('Int64')
    field_codebook['selectable']  = pd.to_numeric(field_codebook['selectable'], errors='coerce').fillna(0).astype('Int8')

    cols_to_type = ['title','coding','meaning','encoding_system']
    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        # Add all acr_abb columns
        acr_cols = [c for c in field_codebook.columns if c.startswith('acr_abb_')]
        cols_to_type.extend(acr_cols)
    else:
        if 'acr_abb' in field_codebook.columns:
            cols_to_type.append('acr_abb')

    for col in cols_to_type:
        if col in field_codebook.columns:
            field_codebook[col] = field_codebook[col].astype('string')

    field_codebook = field_codebook.loc[:, ~field_codebook.columns.duplicated()]

    # ---------------- save outputs ----------------
    print("\n💾 Saving output files...")
    # Enriched fields
    parquet_path = output_dir / 'ukb_fields.parquet'
    csv_path = output_dir / 'ukb_fields.csv'
    enriched.to_parquet(parquet_path, index=False, compression='snappy')
    enriched.to_csv(csv_path, index=False)
    print(f"  ✓ Saved fields: {parquet_path.name}, {csv_path.name}")

    # Field codebook
    cb_parquet = output_dir / 'ukb_field_codebook.parquet'
    cb_csv = output_dir / 'ukb_field_codebook.csv'
    field_codebook.to_parquet(cb_parquet, index=False, compression='snappy')
    field_codebook.to_csv(cb_csv, index=False)
    print(f"  ✓ Saved codebook: {cb_parquet.name}, {cb_csv.name}")

    # dtype + validation report
    dtype_path = output_dir / 'column_dtypes.txt'
    with open(dtype_path, 'w') as f:
        f.write("Column Data Types\n")
        f.write("=" * 50 + "\n\n")
        f.write("FIELD.TXT COLUMNS:\n")
        for col, dtype in field_types.items():
            f.write(f"  {col}: {dtype}\n")
        f.write("\nCATEGORY.TXT COLUMNS:\n")
        for col, dtype in cat_types.items():
            f.write(f"  {col}: {dtype}\n")
        f.write("\nENCODING.TXT COLUMNS:\n")
        for col, dtype in enc_types.items():
            f.write(f"  {col}: {dtype}\n")
        f.write("\nFINAL FIELDS COLUMNS:\n")
        for col in enriched.columns:
            f.write(f"  {col}: {enriched[col].dtype}\n")
        f.write("\nFIELD CODEBOOK COLUMNS:\n")
        for col in field_codebook.columns:
            f.write(f"  {col}: {field_codebook[col].dtype}\n")
    print(f"  ✓ Saved dtype mapping: {dtype_path.name}")

    # Statistics
    total_fields = len(enriched)
    matched_fields = ((enriched['category_title'] != 'No Category Assigned') &
                      (enriched['category_title'] != 'Unmatched Category')).sum()
    no_category_fields = (enriched['category_title'] == 'No Category Assigned').sum()
    unmatched_fields = (enriched['category_title'] == 'Unmatched Category').sum()

    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        abbr_matched = enriched['acr_abb_1'].fillna("").astype(str).str.len().gt(0).sum()
    else:
        abbr_matched = enriched.get('acr_abb', pd.Series([""] * len(enriched), dtype='string')).fillna("").astype(str).str.len().gt(0).sum()
    
    # Analyze acronym coverage
    print("\n📊 ACRONYM COVERAGE ANALYSIS")
    print("=" * 50)

    # Count unique acronyms captured
    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        has_acronym = enriched['acr_abb_1'].fillna("").astype(str).str.len() > 0
        matched_df = enriched[has_acronym].copy()
        
        unique_acronyms = set()
        for col in [c for c in enriched.columns if c.startswith('acr_abb_')]:
            for title in matched_df[col]:
                if pd.notna(title) and title:
                    acronyms = re.findall(r'\b[A-Z]{2,10}\b', str(title))
                    unique_acronyms.update(acronyms)
        
        # ADD THE MISSING STATS OUTPUT:
        print(f"✓ Fields with replacements: {len(matched_df):,} / {total_fields:,} ({len(matched_df)/total_fields*100:.1f}%)")
        print(f"✓ Unique acronyms used: {len(unique_acronyms)}")
        print(f"✓ Top 20 acronyms: {sorted(list(unique_acronyms))[:20]}")
        
        no_acronym = enriched[~has_acronym].copy()
        print(f"\n✗ Fields without acronyms: {len(no_acronym):,}")

        print("\nPotential missed opportunities (sample):")
        for _, row in no_acronym.head(20).iterrows():
            title = row['title']
            parens = re.findall(r'\(([A-Z]{2,10})\)', str(title))
            if parens:
                print(f"  - '{title}' (contains: {parens})")
        
    else:  # Single-column mode
        if 'acr_abb' in enriched.columns:
            has_acronym = enriched['acr_abb'].fillna("").astype(str).str.len() > 0
            matched_df = enriched[has_acronym].copy()  # ADD THIS LINE
            
            unique_acronyms = set()
            for title in matched_df['acr_abb']:
                acronyms = re.findall(r'\b[A-Z]{2,10}\b', str(title))
                unique_acronyms.update(acronyms)
            
            print(f"✓ Fields with replacements: {len(matched_df):,} / {total_fields:,} ({len(matched_df)/total_fields*100:.1f}%)")
            print(f"✓ Unique acronyms used: {len(unique_acronyms)}")
            print(f"✓ Top 20 acronyms: {sorted(list(unique_acronyms))[:20]}")
            
            no_acronym = enriched[~has_acronym].copy()
            print(f"\n✗ Fields without acronyms: {len(no_acronym):,}")
            
            print("\nPotential missed opportunities (sample):")
            for _, row in no_acronym.head(20).iterrows():
                title = row['title']
                parens = re.findall(r'\(([A-Z]{2,10})\)', str(title))
                if parens:
                    print(f"  - '{title}' (contains: {parens})")
        else:
            # No acronym column exists
            print("✗ No acronym columns found")

    print("\n📝 Exporting unique acronym mappings...")
    unique_mappings = {}
    
    # Get candidates_df from tracking if available
    candidates_df = pd.DataFrame()
    candidates_file = output_dir / 'acronym_candidates.csv'
    if candidates_file.exists():
        candidates_df = pd.read_csv(candidates_file)

    if not candidates_df.empty:
        chosen_only = candidates_df[candidates_df['chosen'] == True].copy()
        
        for _, row in chosen_only.iterrows():
            phrase = row['matched_phrase']
            acronym = row['candidate_acronym']
            if pd.notna(phrase) and pd.notna(acronym):
                normalized = phrase.lower().strip()
                if normalized not in unique_mappings:
                    unique_mappings[normalized] = acronym
                elif unique_mappings[normalized] != acronym:
                    print(f"  Conflict: '{phrase}' maps to both '{unique_mappings[normalized]}' and '{acronym}'")

    sorted_mappings = dict(sorted(unique_mappings.items()))

    # Write acronym mappings
    mappings_file = output_dir / 'acronym_mappings.txt'
    with open(mappings_file, 'w') as f:
        f.write("UNIQUE ACRONYM MAPPINGS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total unique mappings: {len(sorted_mappings)}\n")
        f.write("=" * 50 + "\n\n")
        
        for phrase, acronym in sorted_mappings.items():
            f.write(f"{phrase} = {acronym}\n")

    print(f"✓ Exported {len(sorted_mappings)} unique mappings to {mappings_file.name}")

    # Create reverse mapping
    acronym_to_phrases = {}
    for phrase, acronym in sorted_mappings.items():
        if acronym not in acronym_to_phrases:
            acronym_to_phrases[acronym] = []
        acronym_to_phrases[acronym].append(phrase)

    sorted_by_acronym = dict(sorted(acronym_to_phrases.items()))

    reverse_file = output_dir / 'acronym_summary.txt'
    with open(reverse_file, 'w') as f:
        f.write("ACRONYMS BY FREQUENCY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total unique acronyms: {len(sorted_by_acronym)}\n")
        f.write("=" * 50 + "\n\n")
        
        for acronym, phrases in sorted(sorted_by_acronym.items(), 
                                    key=lambda x: len(x[1]), reverse=True):
            f.write(f"\n{acronym} ({len(phrases)} phrases):\n")
            for phrase in sorted(phrases)[:5]:
                f.write(f"  - {phrase}\n")
            if len(phrases) > 5:
                f.write(f"  ... and {len(phrases)-5} more\n")

    print(f"✓ Exported acronym summary to {reverse_file.name}")

    # Validation report
    report_path = output_dir / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("UK Biobank Data Processing Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("CATEGORY LINKAGE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total fields processed: {total_fields:,}\n")
        f.write(f"Fully matched fields: {matched_fields:,} ({matched_fields/total_fields*100:.1f}%)\n")
        f.write(f"No category (null/empty): {no_category_fields:,} ({no_category_fields/total_fields*100:.1f}%)\n")
        f.write(f"Unmatched category IDs: {unmatched_fields:,} ({unmatched_fields/total_fields*100:.1f}%)\n\n")
        f.write("ABBREVIATION MATCHING (Meta-Inventory)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fields with Meta-Inventory abbreviations: {abbr_matched:,} ({abbr_matched/total_fields*100:.1f}%)\n\n")
        f.write("DICTIONARY COVERAGE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Unique encodings referenced by fields: {len(enc_used)}\n")
        f.write(f"Dictionary rows available: {len(dictionary_values):,}\n")
        f.write("\nEncoding IDs used (with system guess):\n")
        enc_used_list = enc_keep[enc_keep['encoding_id'].isin(enc_used)][['encoding_id','encoding_title','encoding_system','encoding_num_members']]
        f.write(enc_used_list.to_string(index=False))
        f.write("\n")
    print(f"  ✓ Saved validation report: {report_path.name}")

    # Generate human verification sample
    print("\n📋 Generating human verification sample...")

    if isinstance(META_INVENTORY_CFG['MAX_ACRONYMS_PER_TITLE'], (list, tuple)):
        matched_acronyms = enriched[enriched['acr_abb_1'].fillna("").astype(str).str.len() > 0].copy()
        # For verification, you might want to use the first column as reference
        matched_acronyms['acr_abb'] = matched_acronyms['acr_abb_1']  # Create temp column for compatibility
    else:
        matched_acronyms = enriched[enriched['acr_abb'].fillna("").astype(str).str.len() > 0].copy()

    if len(matched_acronyms) > 0:
        if candidates_file.exists():
            candidates_data = pd.read_csv(candidates_file)
            chosen_replacements = candidates_data[candidates_data['chosen'] == True][
                ['field_id', 'matched_phrase', 'candidate_acronym']
            ].drop_duplicates()
            
            matched_acronyms = matched_acronyms.merge(
                chosen_replacements, 
                on='field_id', 
                how='left'
            )
        else:
            matched_acronyms['matched_phrase'] = ""
            matched_acronyms['candidate_acronym'] = ""
        
        matched_acronyms['replacement_key'] = (
            matched_acronyms['matched_phrase'].fillna('') + ' → ' + 
            matched_acronyms['candidate_acronym'].fillna('')
        )
        
        unique_replacements = matched_acronyms.groupby('replacement_key').agg({
            'field_id': 'first',
            'title': 'first',
            'acr_abb': 'first',
            'matched_phrase': 'first',
            'candidate_acronym': 'first',
            'replacement_key': 'count'
        }).rename(columns={'replacement_key': 'occurrence_count'}).reset_index()
        
        print(f"  Original matches: {len(matched_acronyms):,}")
        print(f"  Unique replacements: {len(unique_replacements):,}")
        print(f"  Deduplication ratio: {len(unique_replacements)/len(matched_acronyms)*100:.1f}%")
        
        total_unique = len(unique_replacements)
        
        import math
        confidence_level = 0.95
        margin_of_error = 0.05
        z_score = 1.96
        p = 0.5
        
        sample_size = math.ceil((z_score**2 * p * (1-p)) / (margin_of_error**2))
        
        if sample_size > total_unique:
            sample_size = total_unique
        elif total_unique > 5000:
            sample_size = math.ceil(sample_size / (1 + ((sample_size - 1) / total_unique)))
        
        print(f"  Statistical sample needed: {sample_size:,} unique replacements")
        
        verification_sample = unique_replacements.sample(
            n=min(sample_size, len(unique_replacements)), random_state=42
        )
        
        verification_sample = verification_sample.sort_values('occurrence_count', ascending=False)
        
        # Create human verification file
        verify_path = output_dir / 'human_verify.txt'
        with open(verify_path, 'w') as f:
            f.write(f"Human Verification Sample for Meta-Inventory Abbreviation Matching\n")
            f.write(f"Total matched fields: {len(matched_acronyms):,}\n")
            f.write(f"Unique replacements: {total_unique:,}\n")
            f.write(f"Sample size for 95% confidence (±5% error): {sample_size:,}\n")
            f.write(f"Actual sample size: {len(verification_sample):,}\n")
            f.write("=" * 80 + "\n")
            f.write("Columns: # | Occurrences | Original Title | Modified Title | What Changed\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("#\tCount\tOriginal Title\tModified Title\tWhat Changed\n")
            f.write("-\t-----\t" + "-"*50 + "\t" + "-"*50 + "\t" + "-"*30 + "\n")
            
            for idx, (_, row) in enumerate(verification_sample.iterrows(), 1):
                original = row['title']
                modified = row['acr_abb']
                occurrences = row['occurrence_count']
                
                if pd.notna(row['matched_phrase']) and pd.notna(row['candidate_acronym']):
                    phrase = str(row['matched_phrase'])
                    acronym = str(row['candidate_acronym'])
                    
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    highlighted_original = pattern.sub(f"▶{phrase}◀", original)
                    highlighted_modified = modified.replace(acronym, f"★{acronym}★") if modified else original
                    change_summary = f"{phrase} → {acronym}"
                else:
                    highlighted_original = original
                    highlighted_modified = modified if modified else original
                    change_summary = "Unknown change"
                
                f.write(f"{idx}\t{occurrences}x\t{highlighted_original}\t{highlighted_modified}\t{change_summary}\n")
        
        print(f"  ✓ Created deduplicated verification sample: {verify_path.name}")
        print(f"  ✓ Verifying {len(verification_sample):,} unique replacements")
        print(f"  ✓ These represent {verification_sample['occurrence_count'].sum():,} total applications")

    print("\n✅ Processing complete! Check ./output for files.")
    print("Inserted columns after encoding_id:", to_place)
    print(f"Added acr_abb column with {abbr_matched}/{total_fields} abbreviations matched")
    print("Created ukb_field_codebook.* mapping fields to all ICD10/ICD9/OPCS/self-report codes.")

if __name__ == "__main__":
    main()
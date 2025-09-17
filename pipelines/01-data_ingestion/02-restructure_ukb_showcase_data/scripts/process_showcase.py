#!/usr/bin/env python3
"""
UK Biobank Field + Category + Dictionary Linker
- Enrich field.txt with dictionary metadata inserted immediately after encoding_id
- Build an exploded codebook linking each field_id to all dictionary codes (ICD10/ICD9/OPCS4/etc.)
- NEW: Add acr_abb column with MRCONSO abbreviation matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
try:
    import ahocorasick  # pip install pyahocorasick
except Exception:
    ahocorasick = None


USE_COMPREHENSIVE_TRACKING = True
# ----------------------------- Acronym scoring UMLS -----------------------------

CLINICAL_STY_PRIORITY = {
    'T059': 10,  # Laboratory Procedure
    'T060': 10,  # Diagnostic Procedure  
    'T034': 9,   # Laboratory or Test Result
    'T201': 9,   # Clinical Attribute
    'T033': 8,   # Finding
    'T047': 8,   # Disease or Syndrome
    'T116': 7,   # Amino Acid, Peptide, or Protein
    'T123': 7,   # Biologically Active Substance
    'T126': 6,   # Enzyme
    'T028': 3,   # Gene or Genome
}

SAB_PRIORITY = {
    'LNC': 200,      # LOINC
    'SNOMEDCT_US': 90,
    'MSH': 80,       # MeSH
    'NCI': 70,
    'RXNORM': 60,
    'UNIPROT': 50,   # UniProt for proteins
    'HGNC': 20,      # Gene sources (lower priority in clinical)
    'NCBI': 15,
    'ENSEMBL': 10,
}

# ----------------------------- Acronym Matching Hyperparameters -----------------------------
# Start LOOSE; tighten by flipping flags or narrowing ranges.
ACRONYM_CFG = {
    # ACRONYM validation (what counts as valid acronym)
    "ACRONYM_MIN_LEN": 2,         # BP, HR are valid acronyms
    "ACRONYM_MAX_LEN": 10,         # SARS-CoV-2 is valid
    
    # PHRASE matching (what phrases we search for)
    "PHRASE_MIN_LEN": 4,          # Don't match "l u" from "ultrasound"
    "PHRASE_MAX_LEN": None,       # No upper limit on phrase length

    # Other
    "ALLOW_HYPHEN": True,         # e.g., NT-proBNP
    "ALLOW_PERIOD": False,         # e.g., T.S.H.
    "ALLOW_SLASH": True,          # e.g., ECG/EKG

    # Case patterns (NEW)
    "ALLOW_FIRST_LOWER": True,  # Allow eGFR, mRNA, pH patterns
    "ALLOW_MIXED_CASE": False,  # Still reject random MiXeD case
    "ALLOW_TRAILING_PLURAL": True,# allow HDLs
    "ALLOW_GREEK": True,         # set True for tokens like TNF-α (needs Unicode regex)

    # Semantic filters
    "LOWERCASE_LINKING_WORDS": {  # Words that are lowercase in titles = not acronyms
    "in", "on", "or", "as", "it", "of", "to", "at", "by", "for", "and", "the", "with"
    },
    "AMBIGUOUS_UPPERCASE": {      # Could be acronyms OR regular words, need context
        "OR",   # Operating Room vs logical OR
        "US",   # Ultrasound vs United States
        "NO",   # Nitric Oxide vs negation
        "IT",   # Information Technology vs pronoun
    },

    # UMLS source/type policy
    "TTY_WHITELIST": {"ACR","AB","AA"},  # UMLS “acronym-like” term types
    "REQUIRE_TTY": True,         # Loose: accept any token that *looks* like acronym shape.
                                  # Tighten: set True to require TTY in whitelist.

    # Phrase-key policy for full names (left side of mapping)
    "REQUIRE_MULTIWORD_FULLNAME": True,  # keep True to avoid single-word like "Sex"
    "ALLOW_DEPAREN": True,               # also index de-parenthesized variants of STR

    # NEW: configurable UMLS governance (no hard-coded lists inside logic)
    "REL_ACCEPTED": ("SY","RQ"),                 # relationship types to consider
    "REL_BONUS": {"SY": 600, "RQ": 300},         # scoring bonus by REL
    "SUPPRESS_EXCLUDE": ("O","Y"),               # drop suppressed/old strings
}
}

# Global switch: substring fallback via Aho–Corasick (loose) vs exact-only (tight)
USE_SUBSTRING_FALLBACK = True  # tighten by setting False
# ========================================================================================


# ----------------------------- helpers you already had -----------------------------

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
                # keep as Int64 to allow NA
                df_typed[col] = df_typed[col].astype('Int64')
        else:
            df_typed[col] = df_str[col]
    return df_typed, dtype_dict

def reorder_columns(df, first_cols):
    all_cols = df.columns.tolist()
    remaining_cols = [col for col in all_cols if col not in first_cols]
    existing_first_cols = [col for col in first_cols if col in all_cols]
    return df[existing_first_cols + remaining_cols]

# ----------------------------- NEW: MRCONSO abbreviation functions (drop-in) -----------------------------

USE_SUBSTRING_FALLBACK = False  # flip to True if you want Aho–Corasick substring hits after exact join

# Minimal, deterministic normalization used on both MRCONSO terms and UKB titles
def _norm(s: str):
    if pd.isna(s) or s is None:
        return None
    s = str(s).lower().strip()
    s = s.replace('-', ' ').replace('/', ' ')
    return ' '.join(s.split())

def is_likely_linking_word(token: str, original_token: str) -> bool:
    """Check if token is a linking word based on original case"""
    # If it was lowercase in original, it's definitely a linking word
    if original_token and original_token[0].islower():
        return True
    
    # If uppercase but in ambiguous list, check context
    if token.upper() in ACRONYM_CFG.get("AMBIGUOUS_UPPERCASE", set()):
        # Could add context checking here
        return False  # For now, assume it's an acronym if uppercase
    
    return False

def make_acronym_validator(cfg=ACRONYM_CFG):
    import re

    allowed = []
    if cfg.get("ALLOW_HYPHEN"): allowed.append("-")
    if cfg.get("ALLOW_SLASH"):  allowed.append("/")
    if cfg.get("ALLOW_PERIOD"): allowed.append(".")
    allowed_class = re.escape("".join(allowed))

    # disallow punctuation that commonly creates false "acronyms"
    banned = [",","+","_","(",")","'",'"',":",";","<",">","?","!","&","\\"]
    if not cfg.get("ALLOW_PERIOD"):
        banned.append(".")
    hard_ban = re.compile("[" + re.escape("".join(banned)) + "]")

    if cfg.get("ALLOW_FIRST_LOWER", True):
        pat = re.compile(rf"^(?:[a-z][A-Z0-9{allowed_class}]+|[A-Z][A-Z0-9{allowed_class}]+)$")
    else:
        pat = re.compile(rf"^[A-Z][A-Z0-9{allowed_class}]+$")

    min_len = cfg.get("ACRONYM_MIN_LEN", 2)
    max_len = cfg.get("ACRONYM_MAX_LEN", 10)

    def is_acronym(s: str) -> bool:
        if not s: return False
        s = s.strip()
        # block spaced separators (prevents "l u" from "ultrasound")
        if " - " in s or " / " in s:
            return False
        # length
        if not (min_len <= len(s) <= max_len):
            return False
        # charset and case
        if hard_ban.search(s):
            return False
        if not pat.fullmatch(s):
            return False
        # optional plural stripping
        if cfg.get("ALLOW_TRAILING_PLURAL") and s.endswith("s") and len(s) > min_len:
            s = s[:-1]
            return bool(pat.fullmatch(s))
        return True

    return is_acronym


# Build a mapping: normalized_fullname -> best_acronym
def build_fullname_to_acronym_map(mrconso_file: Path) -> dict:
    """
    Returns {normalized_english_fullname_variant -> best_english_acronym}.
    """
    usecols = [0,1,2,4,6,11,12,14,16]
    names   = ['CUI','LAT','TS','STT','ISPREF','SAB','TTY','STR','SUPPRESS']

    is_acronym = make_acronym_validator(ACRONYM_CFG)
    TTY_WL = tuple(ACRONYM_CFG["TTY_WHITELIST"])

    def _strip_trailing_paren(s: str) -> str:
        return re.sub(r'\s*\([^)]*\)\s*$', '', s.strip()).strip()

    # Collect ENG rows per CUI - THESE MUST BE OUTSIDE THE LOOP
    cui_fullnames = {}   # CUI -> set(ENG strings)
    cui_best_acr  = {}   # CUI -> (acr_str, rank_tuple)

    def rank(tty_val: str, s: str):
        if ACRONYM_CFG["REQUIRE_TTY"]:
            tty_rank = TTY_WL.index(tty_val) if tty_val in TTY_WL else 999
        else:
            tty_rank = TTY_WL.index(tty_val) if tty_val in TTY_WL else len(TTY_WL)
        return (tty_rank, len(s))

    print(f"\n🔤 Loading MRCONSO from {mrconso_file.name} in chunks...")
    
    chunk_count = 0
    # STEP 1: Process ALL chunks to collect CUI data
    for chunk in pd.read_csv(
        mrconso_file,
        sep='|', header=None, names=names, usecols=usecols,
        dtype=str, chunksize=500_000, on_bad_lines='skip', engine='c'
    ):
        chunk_count += 1
        print(f"  Processing chunk {chunk_count}...")
        
        eng = chunk[
            (chunk.LAT == 'ENG') &
            (~chunk.SUPPRESS.fillna('').isin(ACRONYM_CFG.get('SUPPRESS_EXCLUDE', ())))
        ].copy()
        eng['STR'] = eng['STR'].fillna('').astype(str)
        eng['TTY'] = eng['TTY'].fillna('').astype(str)

        # 1) Gather ALL ENG strings per CUI as potential full-name keys
        for row in eng.itertuples(index=False):
            s = row.STR.strip()
            if not s:
                continue
            cui_fullnames.setdefault(row.CUI, set()).add(s)

        # 2) Evaluate acronym candidates
        for row in eng.itertuples(index=False):
            cand = row.STR.strip()
            if not is_acronym(cand):
                continue
            if ACRONYM_CFG["REQUIRE_TTY"] and row.TTY not in TTY_WL:
                continue

            r = rank(row.TTY or "", cand)
            old = cui_best_acr.get(row.CUI)
            if (old is None) or (r < old[1]):
                cui_best_acr[row.CUI] = (cand, r)

    print(f"  Processed {chunk_count} chunks total")
    
    # STEP 2: Build final mapping AFTER processing all chunks
    print("  Building final fullname→acronym mapping...")
    full2acr = {}
    for cui, (acr, _) in cui_best_acr.items():
        for full in cui_fullnames.get(cui, []):
            if not full:
                continue
            variants = {full}
            if ACRONYM_CFG["ALLOW_DEPAREN"]:
                variants.add(_strip_trailing_paren(full))

            for v in variants:
                v = (v or "").strip()
                if not v:
                    continue
                norm_key = _norm(v)
                if not norm_key:
                    continue
                if ACRONYM_CFG["REQUIRE_MULTIWORD_FULLNAME"] and ' ' not in norm_key:
                    continue
                full2acr[norm_key] = acr

    # STEP 3: Print final statistics
    print(f"  ✓ ENG CUIs with any name: {len(cui_fullnames):,}")
    print(f"  ✓ CUIs with good acronyms: {len(cui_best_acr):,}")
    print(f"  ✓ Fullname→acronym pairs: {len(full2acr):,}")
    
    return full2acr

def build_comprehensive_acronym_data(mrconso_file: Path, mrsty_file: Path = None):
    """
    Returns:
    - full2candidates: {norm_fullname -> [(acr, cui, sab, tty, sty_list, score), ...]}
    - full2best: {norm_fullname -> best_acronym} for backwards compatibility
    """
    is_acronym = make_acronym_validator(ACRONYM_CFG)
    TTY_WL = tuple(ACRONYM_CFG["TTY_WHITELIST"])
    
    # Load semantic types if available
    cui_to_sty = {}
    if mrsty_file and mrsty_file.exists():
        print(f"  Loading semantic types from {mrsty_file.name}...")
        # MRSTY.RRF has trailing pipe delimiter, so we need to handle that
        sty_df = pd.read_csv(mrsty_file, sep='|', header=None,
                            usecols=[0,1,2,3,4,5],  # Keep all 6 columns
                            names=['CUI','TUI','STN','STY','ATUI','CVF'],
                            dtype=str, on_bad_lines='skip')
        for cui, group in sty_df.groupby('CUI'):
            cui_to_sty[cui] = list(group['TUI'].unique())
    
    # Collect all candidates per CUI
    cui_candidates = {}  # CUI -> [(acr, sab, tty), ...]
    cui_fullnames = {}   # CUI -> set(strings)
    
    print(f"\n🔤 Loading MRCONSO from {mrconso_file.name}...")
    for chunk in pd.read_csv(
        mrconso_file, sep='|', header=None,
        names=['CUI','LAT','TS','STT','ISPREF','SAB','TTY','STR'],
        usecols=[0,1,2,4,6,11,12,14], dtype=str, chunksize=500_000
    ):
        eng = chunk[chunk.LAT == 'ENG'].copy()
        
        # Collect all fullnames
        for row in eng.itertuples(index=False):
            if pd.isna(row.STR):
                continue
            s = str(row.STR).strip()
            if s:
                cui_fullnames.setdefault(row.CUI, set()).add(s)
                
        # Collect all acronym candidates (separately!)
        for row in eng.itertuples(index=False):
            if pd.isna(row.STR):
                continue
            cand = str(row.STR).strip()
            
            # Skip linking words
            if is_likely_linking_word(cand.lower(), cand):
                continue
                
            if is_acronym(cand):
                cui_candidates.setdefault(row.CUI, []).append(
                    (cand, row.SAB, row.TTY)
                )
    
    # Build mapping with ALL candidates
    full2candidates = {}
    full2best = {}

    if cui_candidates:  # Only debug if we have data
        debug_specific_terms(cui_candidates, cui_fullnames, cui_to_sty)
    
    for cui, candidates in cui_candidates.items():
        if not candidates:
            continue
            
        sty_list = cui_to_sty.get(cui, [])
        
        # Score each candidate
        scored = []
        for acr, sab, tty in candidates:
            score = compute_acronym_score(acr, sab, tty, sty_list)
            scored.append((acr, cui, sab, tty, sty_list, score))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: (-x[5], len(x[0]), x[0]))  # score, length, alpha
        
        # Map to all fullnames for this CUI
        for full in cui_fullnames.get(cui, []):
            norm_key = _norm(full)
            # ADD THIS: Skip very short phrases that are likely to cause false matches
            if norm_key and len(norm_key) >= ACRONYM_CFG.get("PHRASE_MIN_LEN", 4):
                if not ACRONYM_CFG["REQUIRE_MULTIWORD_FULLNAME"] or ' ' in norm_key:
                    # FILTER: Only keep acronyms that are actually SHORTER than the full phrase
                    # Remove spaces from comparison to handle multi-word phrases properly
                    full_no_spaces = full.replace(' ', '').replace('-', '').replace('/', '')
                    valid_scored = []
                    for candidate in scored:
                        acr = candidate[0]
                        acr_no_spaces = acr.replace(' ', '').replace('-', '').replace('/', '')
                        # Acronym must be shorter than the original phrase
                        if len(acr_no_spaces) < len(full_no_spaces):
                            valid_scored.append(candidate)
                    
                    # Only add if we have valid (shorter) acronyms
                    if valid_scored:
                        full2candidates.setdefault(norm_key, []).extend(valid_scored)
                        if norm_key not in full2best:
                            full2best[norm_key] = valid_scored[0][0]  # best acronym
    
    return full2candidates, full2best

def build_comprehensive_acronym_data_with_relations(mrconso_file: Path, mrrel_file: Path, mrsty_file: Path = None):
    """
    Build acronym mappings using MRREL to link related CUIs
    This catches cases where acronym and full name have different CUIs
    """
    is_acronym = make_acronym_validator(ACRONYM_CFG)
    TTY_WL = tuple(ACRONYM_CFG["TTY_WHITELIST"])
    
    # STEP 1: Load semantic types if available
    cui_to_sty = {}
    if mrsty_file and mrsty_file.exists():
        print(f"  Loading semantic types from {mrsty_file.name}...")
        sty_df = pd.read_csv(mrsty_file, sep='|', header=None,
                            usecols=[0,1], names=['CUI','TUI'],
                            dtype=str, on_bad_lines='skip')
        for cui, group in sty_df.groupby('CUI'):
            cui_to_sty[cui] = list(group['TUI'].unique())
    
    # STEP 2: Load CUI relationships from MRREL
    print(f"\n🔗 Loading CUI relationships from {mrrel_file.name}...")
    # MRREL columns: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
    # We only need CUI1 (col 0) and CUI2 (col 4) for concept linking
    cui_relations = {}  # CUI -> set of related CUIs
    
    for chunk in pd.read_csv(mrrel_file, sep='|', header=None,
                            usecols=[0, 3, 4], names=['CUI1', 'REL', 'CUI2'],
                            dtype=str, chunksize=500_000, on_bad_lines='skip'):
        # Only keep synonym-like relationships
        chunk = chunk[chunk['REL'].isin(['SY', 'RQ', 'RO', 'AQ', 'QB'])]
        for row in chunk.itertuples(index=False):
            # Store the relationship type with the CUI
            cui_relations.setdefault(row.CUI1, {})[row.CUI2] = row.REL
            cui_relations.setdefault(row.CUI2, {})[row.CUI1] = row.REL  # Bidirectional
    
    print(f"  Loaded relationships for {len(cui_relations):,} CUIs")
    
    # STEP 3: Collect acronyms and fullnames from MRCONSO
    print(f"\n🔤 Loading terms from {mrconso_file.name}...")
    all_acronyms = {}  # CUI -> [(acr, sab, tty), ...]
    all_fullnames = {}  # CUI -> set(strings)
    
    # Change this part where you read MRCONSO:
    for chunk in pd.read_csv(
        mrconso_file, sep='|', header=None,
        names=['CUI','LAT','TS','STT','ISPREF','SAB','TTY','STR'],  
        usecols=[0,1,2,4,6,11,12,14], dtype=str, chunksize=500_000
    ):
        eng = chunk[chunk.LAT == 'ENG'].copy()
        
        for row in eng.itertuples(index=False):
            if pd.isna(row.STR):
                continue
            s = str(row.STR).strip()
            
            if is_acronym(s):
                # Store ISPREF with the acronym
                all_acronyms.setdefault(row.CUI, []).append(
                    (s, row.SAB, row.TTY, row.ISPREF)  # <- Added ISPREF
                )
            
            # Also store as potential fullname
            all_fullnames.setdefault(row.CUI, set()).add(s)
    
    print(f"  Found {len(all_acronyms):,} CUIs with acronyms")
    print(f"  Found {len(all_fullnames):,} CUIs with names")
    
    # STEP 4: Build mappings using relationships
    print("\n🔧 Building acronym mappings using CUI relationships...")
    full2candidates = {}
    full2best = {}
    
    # For each CUI with fullnames, look for acronyms in related CUIs
    for full_cui, fullnames in all_fullnames.items():
        # Get all related CUIs with their relationship types
        related_info = cui_relations.get(full_cui, {})
        
        # Collect all acronyms from related CUIs
        combined_acronyms = []
        
        # First add acronyms from the same CUI (highest priority)
        if full_cui in all_acronyms:
            for acr, sab, tty, ispref in all_acronyms[full_cui]:
                sty_list = cui_to_sty.get(full_cui, [])
                score = compute_acronym_score(acr, sab, tty, sty_list, ispref)
                score += 1000  # Bonus for same CUI
                combined_acronyms.append((acr, full_cui, sab, tty, sty_list, score))
        
        # Then add from related CUIs with relationship bonus
        for rel_cui, rel_type in related_info.items():
            if rel_cui in all_acronyms:
                # Relationship type bonus
                rel_bonus = {
                    'SY': 600,   # Direct synonym - highest priority
                    'RQ': 300,   # Related, possibly synonymous
                    'RO': 150,   # Has relationship other than synonymy
                    'AQ': 20,   # Allowed qualifier
                    'QB': 10     # Can be qualified by
                }.get(rel_type, 0)
                
                for acr, sab, tty, ispref in all_acronyms[rel_cui]:
                    sty_list = cui_to_sty.get(rel_cui, [])
                    score = compute_acronym_score(acr, sab, tty, sty_list, ispref)
                    score += rel_bonus  # Add relationship bonus
                    combined_acronyms.append((acr, rel_cui, sab, tty, sty_list, score))
        
        # Loop 
        combined_acronyms.sort(key=lambda x: (-x[5], len(x[0]), x[0]))

        # Map all fullnames to these acronyms
        for full in fullnames:
            norm_key = _norm(full)
            if norm_key and len(norm_key) >= ACRONYM_CFG.get("PHRASE_MIN_LEN", 4):
                if not ACRONYM_CFG["REQUIRE_MULTIWORD_FULLNAME"] or ' ' in norm_key:
                    # Check acronyms are shorter than fullname
                    full_no_spaces = full.replace(' ', '').replace('-', '').replace('/', '')
                    valid_acronyms = []
                    for candidate in combined_acronyms:
                        acr = candidate[0]
                        acr_no_spaces = acr.replace(' ', '').replace('-', '').replace('/', '')
                        if len(acr_no_spaces) < len(full_no_spaces):
                            valid_acronyms.append(candidate)
                    
                    if valid_acronyms:
                        full2candidates.setdefault(norm_key, []).extend(valid_acronyms[:10])  # Limit to top 10
                        if norm_key not in full2best:
                            full2best[norm_key] = valid_acronyms[0][0]
    
    # Deduplicate candidates per fullname
    for norm_key in full2candidates:
        seen = set()
        unique = []
        for cand in full2candidates[norm_key]:
            if cand[0] not in seen:  # Dedupe by acronym text
                seen.add(cand[0])
                unique.append(cand)
        full2candidates[norm_key] = unique
    
    print(f"  ✓ Created {len(full2best):,} fullname→acronym mappings")
    
    # Debug output for problem terms
    print("\n🔍 Checking problem terms:")
    for test in ['aspartate aminotransferase', 'alanine aminotransferase', 'cystatin c']:
        norm_test = _norm(test)
        if norm_test in full2best:
            print(f"  '{test}' → '{full2best[norm_test]}' ✓")
        else:
            print(f"  '{test}' → NOT FOUND ✗")
    
    return full2candidates, full2best

def compute_acronym_score(acr: str, sab: str, tty: str, sty_list: list, ispref: str = 'N') -> float:

    """Compute deterministic score for an acronym candidate"""
    score = 0.0
    
    # TTY bonus
    if tty in ACRONYM_CFG["TTY_WHITELIST"]:
        score += 50
    
    # SAB priority  
    score += SAB_PRIORITY.get(sab, 1)

    # Use ISPREF from UMLS
    if ispref == 'Y':
        score += 300
    
    # STY bonus (prefer clinical/analyte)
    max_sty_score = 0
    for sty in sty_list:
        max_sty_score = max(max_sty_score, CLINICAL_STY_PRIORITY.get(sty, 0))
    score += max_sty_score * 10
    
    # Length penalty (prefer shorter)
    score -= len(acr) * 0.1
    
    return score

def debug_specific_terms(cui_candidates, cui_fullnames, cui_to_sty):
    """Debug specific problematic terms to see what's happening"""
    
    problem_terms = {
        'aspartate aminotransferase': ['AST', 'GOT', 'SGOT'],
        'cystatin c': ['CysC', 'CST3', 'CYSC'],
        'alanine aminotransferase': ['ALT', 'GPT', 'SGPT'],
    }
    
    print("\n🔍 DEBUGGING ACRONYM SELECTION:")
    print("-" * 60)
    
    for cui, fullnames in cui_fullnames.items():
        for full in fullnames:
            full_lower = full.lower()
            for term, expected_acrs in problem_terms.items():
                if term in full_lower:
                    print(f"\nFound '{full}' (CUI: {cui})")
                    
                    if cui in cui_candidates:
                        candidates = cui_candidates[cui]
                        sty_list = cui_to_sty.get(cui, [])
                        
                        print("  Candidates found:")
                        for acr, sab, tty in candidates:
                            score = compute_acronym_score(acr, sab, tty, sty_list)
                            in_expected = "✓" if acr in expected_acrs else "✗"
                            print(f"    {in_expected} {acr:<10} SAB:{sab:<15} TTY:{tty:<5} Score:{score:.1f}")
                    else:
                        print("  No candidates found!")

def apply_acronyms_with_tracking(fields_df: pd.DataFrame, full2candidates: dict) -> tuple:
    """
    Returns (Series of applied acronyms, DataFrame of all candidates)
    Now with SUBSTRING matching like the original!
    """
    if not ahocorasick:
        print("⚠️ pyahocorasick required for substring matching!")
        return pd.Series([""] * len(fields_df), dtype='string'), pd.DataFrame()
    
    # Build Aho-Corasick automaton for substring matching
    A = ahocorasick.Automaton()
    for full_phrase, candidates in full2candidates.items():
        if full_phrase and len(full_phrase) >= ACRONYM_CFG.get("PHRASE_MIN_LEN", 4):
            A.add_word(full_phrase, (full_phrase, candidates))
    A.make_automaton()
    
    applied = []
    all_candidates_list = []
    
    for idx, row in fields_df.iterrows():
        title = row['title']
        field_id = row['field_id']
        
        if pd.isna(title):
            applied.append("")
            continue
        
        norm_title = _norm(str(title))
        if not norm_title:
            applied.append("")
            continue
        
        # Find ALL substring matches in this title
        matches = []
        for end_pos, payload in A.iter(norm_title):
            matched_phrase, candidates = payload
            start_pos = end_pos - len(matched_phrase) + 1
            matches.append((start_pos, end_pos, matched_phrase, candidates))
        
        if not matches:
            applied.append("")
            continue
        
        # Sort by longest match first
        matches.sort(key=lambda x: -(x[1] - x[0] + 1))
        
        # Take the longest non-overlapping match
        chosen = None
        for start, end, phrase, candidates in matches:
            # For now, just take the first/longest match
            if not chosen:
                chosen = (phrase, candidates)
                break
        
        if not chosen:
            applied.append("")
            continue
        
        matched_phrase, candidates = chosen
        
        # Check for conflicts in the top candidates
        has_gene = any('T028' in c[4] for c in candidates[:5] if len(c) > 4)
        has_clinical = any(
            any(sty in CLINICAL_STY_PRIORITY for sty in c[4]) 
            for c in candidates[:5] if len(c) > 4
        )
        conflict = has_gene and has_clinical
        
        # Choose best and apply the replacement
        best = candidates[0]
        
        # Actually replace the phrase in the title
        import re
        tokens = [re.escape(tok) for tok in matched_phrase.split()]


        if len(matched_phrase) <= 3 and ' ' not in matched_phrase:
            # For very short phrases without spaces, ensure they're whole words
            # This prevents "l u" from inside "ultrasound"
            continue  # Skip this match entirely
            
        pattern = re.compile(r'\b' + r'[\s\-/]+'.join(tokens) + r'\b', re.IGNORECASE)

        # Check if this creates a redundant pattern like "BMD (BMD)"
        # If the acronym already appears before the parentheses, remove the whole parenthetical
        acronym = best[0]
        redundant_pattern = re.compile(
            rf'\b{re.escape(acronym)}\s*\(\s*{re.escape(acronym)}\s*\)', 
            re.IGNORECASE
        )

        replaced_title = pattern.sub(acronym, title)

        # Fix redundant acronyms like "BMD (BMD)" -> "BMD"
        if redundant_pattern.search(replaced_title):
            replaced_title = redundant_pattern.sub(acronym, replaced_title)

        # Also handle case where acronym is already there: "BMD (bone mineral density)" -> "BMD"
        already_there_pattern = re.compile(
            rf'\b{re.escape(acronym)}\s*\(\s*{re.escape(matched_phrase)}\s*\)',
            re.IGNORECASE
        )
        if already_there_pattern.search(title):
            replaced_title = already_there_pattern.sub(acronym, title)
        
        # Only count as "applied" if we actually changed something
        if replaced_title != title:
            applied.append(replaced_title)
        else:
            applied.append("")
        
        # Record all candidates for this match
        for rank, cand in enumerate(candidates[:10], 1):
            if len(cand) >= 6:  # Full candidate tuple
                acr, cui, sab, tty, sty_list, score = cand
                all_candidates_list.append({
                    'field_id': field_id,
                    'title': title,
                    'matched_phrase': matched_phrase,
                    'replaced_title': replaced_title if replaced_title != title else "",
                    'candidate_acronym': acr,
                    'cui': cui,
                    'sab': sab,
                    'tty': tty,
                    'sty': '|'.join(sty_list) if sty_list else "",
                    'score': score,
                    'rank': rank,
                    'chosen': rank == 1,
                    'has_conflict': conflict,
                    'domain': 'gene' if 'T028' in sty_list else 'clinical'
                })
    
    candidates_df = pd.DataFrame(all_candidates_list)
    return pd.Series(applied, dtype='string'), candidates_df

def debug_blood_pressure_mapping(full2acr):
    print("\n🔍 DEBUGGING: Blood pressure mapping")
    
    # Check all keys containing "blood pressure"
    bp_keys = [k for k in full2acr.keys() if 'blood pressure' in k.lower()]
    print(f"Keys containing 'blood pressure': {len(bp_keys)}")
    for key in bp_keys[:10]:  # show first 10
        print(f"  '{key}' → '{full2acr[key]}'")
    
    # Check if normalized "blood pressure" exists
    norm_bp = _norm("blood pressure")
    print(f"Normalized 'blood pressure': '{norm_bp}'")
    if norm_bp in full2acr:
        print(f"✓ Found: '{norm_bp}' → '{full2acr[norm_bp]}'")
    else:
        print(f"✗ Missing: '{norm_bp}' not found in mapping")
        
    # Check what maps to 'BP'
    bp_mappings = [(k,v) for k,v in full2acr.items() if v == 'BP']
    print(f"Terms that map to 'BP': {len(bp_mappings)}")
    for key, val in bp_mappings[:10]:
        print(f"  '{key}' → '{val}'")
    
    # Check if BP appears as any value
    all_bp_values = [(k,v) for k,v in full2acr.items() if 'BP' in v]
    print(f"Terms with 'BP' in the value: {len(all_bp_values)}")
    for key, val in all_bp_values[:10]:
        print(f"  '{key}' → '{val}'")

def debug_specific_replacements(fields_df: pd.DataFrame, full2acr: dict, debug_terms=None):
    """Debug specific term replacements to see what's happening"""
    if debug_terms is None:
        debug_terms = ["blood pressure", "systolic blood pressure", "diastolic blood pressure"]
    
    print(f"\n🔍 DEBUGGING SPECIFIC REPLACEMENTS")
    print("-" * 50)
    
    # Find fields containing our debug terms
    for term in debug_terms:
        print(f"\nSearching for fields containing: '{term}'")
        
        # Find exact matches
        exact_matches = fields_df[fields_df['title'].str.lower() == term.lower()]
        print(f"  Exact matches: {len(exact_matches)}")
        for idx, row in exact_matches.head(3).iterrows():
            print(f"    '{row['title']}'")
        
        # Find partial matches  
        partial_matches = fields_df[fields_df['title'].str.lower().str.contains(term.lower(), na=False)]
        print(f"  Partial matches: {len(partial_matches)}")
        for idx, row in partial_matches.head(5).iterrows():
            print(f"    '{row['title']}'")
            
        # Test the replacement logic on these specific cases
        norm_term = _norm(term)
        if norm_term in full2acr:
            expected_replacement = full2acr[norm_term]
            print(f"  Expected: '{term}' → '{expected_replacement}'")
            
            # Test replacement on sample titles
            for idx, row in partial_matches.head(3).iterrows():
                title = row['title']
                norm_title = _norm(title)
                print(f"    Testing: '{title}'")
                print(f"    Normalized: '{norm_title}'")
                
                # Check if our term appears in normalized title
                if norm_term in norm_title:
                    print(f"    ✓ Contains normalized term")
                    # Simulate the replacement
                    test_result = norm_title.replace(norm_term, expected_replacement)
                    print(f"    Test replacement: '{test_result}'")
                else:
                    print(f"    ✗ Does NOT contain normalized term")
        else:
            print(f"  ✗ '{norm_term}' not found in mapping")

def replace_fullnames_with_acronyms_debug(fields_df: pd.DataFrame, full2acr: dict) -> pd.Series:
    """Enhanced version with debugging for blood pressure specifically"""
    
    if ahocorasick is None:
        print("⚠️ pyahocorasick not available; using simple mapping.")
        titles_norm = fields_df['title'].map(_norm)
        mapped = titles_norm.map(full2acr)
        return mapped.fillna("").astype('string')

    def _norm_local(s: str):
        if pd.isna(s) or s is None:
            return None
        s = str(s).lower().strip()
        s = s.replace('-', ' ').replace('/', ' ')
        return ' '.join(s.split())

    # Build automaton
    A = ahocorasick.Automaton()
    for full_n, acr in full2acr.items():
        if full_n and len(full_n) >= 3:  # This might be the issue!
            A.add_word(full_n, (full_n, acr))
    A.make_automaton()

    out = []
    debug_bp_count = 0
    
    for i, title in enumerate(fields_df['title']):
        if pd.isna(title):
            out.append("")
            continue

        raw = str(title)
        norm_title = _norm_local(raw)
        if not norm_title:
            out.append("")
            continue

        # Debug blood pressure specifically
        is_bp_related = 'blood pressure' in norm_title.lower()
        if is_bp_related and debug_bp_count < 5:
            print(f"\n🩺 BP DEBUG #{debug_bp_count + 1}: '{raw}'")
            print(f"    Normalized: '{norm_title}'")
            debug_bp_count += 1

        matches = []
        for end, payload in A.iter(norm_title):
            full_n, acr = payload
            start = end - len(full_n) + 1
            matches.append((start, end, full_n, acr))
            
            if is_bp_related and debug_bp_count <= 5:
                print(f"    Match found: '{full_n}' → '{acr}' at positions {start}-{end}")

        if not matches:
            if is_bp_related and debug_bp_count <= 5:
                print(f"    No matches found!")
            out.append("")
            continue

        # Sort by match length (longest first) and position
        matches.sort(key=lambda x: (-(x[1]-x[0]+1), x[0]))
        
        if is_bp_related and debug_bp_count <= 5:
            print(f"    Sorted matches: {[(m[2], m[3]) for m in matches]}")

        # Resolve overlapping matches
        chosen = []
        taken = [False] * len(norm_title)
        for s, e, full_n, acr in matches:
            if any(taken[i] for i in range(s, e+1)):
                continue
            for i in range(s, e+1):
                taken[i] = True
            chosen.append((s, e, full_n, acr))

        if is_bp_related and debug_bp_count <= 5:
            print(f"    Chosen matches: {[(m[2], m[3]) for m in chosen]}")

        # Apply replacements
        result = raw
        changed = False
        for _, _, full_n, acr in sorted(chosen, key=lambda x: -(x[1]-x[0]+1)):
            tokens = [re.escape(tok) for tok in full_n.split()]
            sep = r'[\s\-/]+'
            pattern = re.compile(r'\b' + sep.join(tokens) + r'\b', flags=re.IGNORECASE)
            
            if is_bp_related and debug_bp_count <= 5:
                print(f"    Applying pattern: {pattern.pattern}")
                print(f"    To replace '{full_n}' → '{acr}'")
            
            new_result, nsubs = pattern.subn(acr, result)
            if nsubs > 0:
                changed = True
                result = new_result
                if is_bp_related and debug_bp_count <= 5:
                    print(f"    ✓ Replaced {nsubs} times: '{result}'")
            elif is_bp_related and debug_bp_count <= 5:
                print(f"    ✗ No replacements made")

        final_result = result if changed and result.strip() != raw.strip() else ""
        if is_bp_related and debug_bp_count <= 5:
            print(f"    Final result: '{final_result}'")
        
        out.append(final_result)
    
    return pd.Series(out, dtype='string')

def replace_fullnames_with_acronyms(fields_df: pd.DataFrame, full2acr: dict) -> pd.Series:
    if ahocorasick is None:
        print("⚠️ pyahocorasick not available; skipping substring replacement.")
        titles_norm = fields_df['title'].map(_norm)
        # exact map; if no hit leave as empty string (not the original)
        mapped = titles_norm.map(full2acr)
        return mapped.fillna("").astype('string')

    def _norm_local(s: str):
        if pd.isna(s) or s is None:
            return None
        s = str(s).lower().strip()
        s = s.replace('-', ' ').replace('/', ' ')
        return ' '.join(s.split())

    A = ahocorasick.Automaton()
    for full_n, acr in full2acr.items():
        if full_n and len(full_n) >= 3:
            A.add_word(full_n, (full_n, acr))
    A.make_automaton()

    out = []
    for title in fields_df['title']:
        if pd.isna(title):
            out.append("")
            continue

        raw = str(title)
        norm_title = _norm_local(raw)
        if not norm_title:
            out.append("")
            continue

        matches = []
        for end, payload in A.iter(norm_title):
            full_n, acr = payload
            start = end - len(full_n) + 1
            matches.append((start, end, full_n, acr))

        if not matches:
            out.append("")  # ← **no acronym replacement → empty string**
            continue

        matches.sort(key=lambda x: (-(x[1]-x[0]+1), x[0]))
        chosen = []
        taken = [False] * len(norm_title)
        for s, e, full_n, acr in matches:
            if any(taken[i] for i in range(s, e+1)):
                continue
            for i in range(s, e+1):
                taken[i] = True
            chosen.append((s, e, full_n, acr))

        result = raw
        changed = False
        for _, _, full_n, acr in sorted(chosen, key=lambda x: -(x[1]-x[0]+1)):
            tokens = [re.escape(tok) for tok in full_n.split()]
            sep = r'[\s\-/]+'
            pattern = re.compile(r'\b' + sep.join(tokens) + r'\b', flags=re.IGNORECASE)
            new_result, nsubs = pattern.subn(acr, result)
            if nsubs > 0:
                changed = True
                result = new_result

        out.append(result if changed and result.strip() != raw.strip() else "")
    return pd.Series(out, dtype='string')

# Primary fast path: exact normalized join
def map_titles_to_abbrev_exact(fields_df: pd.DataFrame, full2acr: dict) -> pd.Series:
    titles_norm = fields_df['title'].map(_norm)
    return titles_norm.map(full2acr).fillna("").astype('string')

# Optional substring fallback via Aho–Corasick over normalized fullnames
def map_titles_to_abbrev_substring(fields_df: pd.DataFrame, full2acr: dict) -> pd.Series:
    if not USE_SUBSTRING_FALLBACK:
        return pd.Series([""] * len(fields_df), dtype='string')

    if ahocorasick is None:
        print("⚠️ Substring fallback requested but pyahocorasick is not available. Skipping.")
        return pd.Series([pd.NA] * len(fields_df), dtype='string')

    A = ahocorasick.Automaton()
    # store full_name as key, acronym as value
    for full_n, acr in full2acr.items():
        if full_n:
            A.add_word(full_n, (full_n, acr))
    A.make_automaton()

    out = []
    for title in fields_df['title']:
        s = _norm(title)
        if not s:
            out.append(pd.NA)
            continue
        best = None  # (span_len, acr)
        for end, payload in A.iter(s):
            full_n, acr = payload
            span_len = len(full_n)
            if (best is None) or (span_len > best[0]):
                best = (span_len, acr)
        out.append(best[1] if best else pd.NA)
    return pd.Series([v if v is not None else "" for v in out], dtype='string')

# ----------------------------- new helpers for dictionaries -----------------------------

ENC_SYSTEM_MAP = {
    # clinical code systems
    19: "ICD10",
    87: "ICD9",
    240: "OPCS4",
    # common self-report / others
    6: "NonCancerIllness",
    5: "Operation",
    4: "Treatment",
    3: "Cancer",
    2: "Employment",
    1: "YesTruePresence",
    7: "YesNo",
    8: "CalendarMonth",
    # catch-all handled later as "Other"
}

def is_hierarchical(structure_value: str) -> bool:
    # Showcase uses 1 = flat, 2 = tree (hierarchical). Treat unknowns defensively.
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

    # force string codes (ICD, OPCS, Read, dates/times, etc.)
    out['coding'] = out['coding'].astype(str)

    # selectable -> 0/1
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

    # force string codes to avoid Arrow int coercion
    out['coding']   = out['coding'].astype(str)
    out['meaning']  = out['meaning'].astype(str)
    out['node_id']  = pd.Series([pd.NA]*len(out), dtype='Int64')
    out['parent_id']= pd.Series([pd.NA]*len(out), dtype='Int64')
    out['selectable']= 1

    return out[['encoding_id','coding','meaning','node_id','parent_id','selectable']]

def move_after(df: pd.DataFrame, anchor_col: str, cols_to_move: list) -> pd.DataFrame:
    """Reorder existing columns so cols_to_move appear immediately after anchor_col."""
    # keep only columns that actually exist and are not the anchor
    cols_to_move = [c for c in cols_to_move if c in df.columns and c != anchor_col]
    if not cols_to_move:
        return df

    # current columns excluding the ones we want to move
    base_cols = [c for c in df.columns if c not in cols_to_move]
    try:
        idx = base_cols.index(anchor_col)
    except ValueError:
        # anchor not found → append moved cols at end (no duplicates)
        return df[base_cols + cols_to_move]

    new_order = base_cols[:idx+1] + cols_to_move + base_cols[idx+1:]
    return df[new_order]

# ----------------------------- main -----------------------------

def main():
    print("🔬 UK Biobank Field / Category / Dictionary Linker")
    print("=" * 50)

    input_dir = Path('/input')
    output_dir = Path('/output')
    output_dir.mkdir(exist_ok=True)

    # core files
    field_file = input_dir / 'field.txt'
    category_file = input_dir / 'category.txt'

    # dictionary index + values
    encoding_file     = input_dir / 'encoding.txt'
    enhierstring_file = input_dir / 'enhierstring.txt'
    enhierint_file    = input_dir / 'enhierint.txt'
    esimpint_file     = input_dir / 'esimpint.txt'
    esimpstring_file  = input_dir / 'esimpstring.txt'
    esimpreal_file    = input_dir / 'esimpreal.txt'
    esimpdate_file    = input_dir / 'esimpdate.txt'
    esimptime_file    = input_dir / 'esimptime.txt'
    
    # NEW: MRCONSO file
    mrconso_file = input_dir / 'MRCONSO.RRF'

    # ---------- validate presence ----------
    missing = [p.name for p in [field_file, category_file, encoding_file] if not p.exists()]
    if missing:
        print(f"❌ ERROR: missing required file(s): {missing}")
        sys.exit(1)

    # optional values files (some datasets won't have all)
    value_files = [f for f in [enhierstring_file, enhierint_file, esimpint_file, esimpstring_file, esimpreal_file, esimpdate_file, esimptime_file] if f.exists()]
    if not value_files:
        print("❌ ERROR: No dictionary value files found. Expected at least one of enhierstring/enhierint/esimp*.txt")
        sys.exit(1)

    # ---------- load core ----------
    fields_df, field_types = load_and_type_dataframe(field_file)
    categories_df, cat_types = load_and_type_dataframe(category_file)
    encoding_df, enc_types = load_and_type_dataframe(encoding_file)

    # ---------- NEW: Load MRCONSO abbreviations if available ----------
    if mrconso_file.exists():
        mrsty_file = input_dir / 'MRSTY.RRF'
        
        if USE_COMPREHENSIVE_TRACKING:
            mrrel_file = input_dir / 'MRREL.RRF'
            if mrrel_file.exists():
                full2candidates, full2acr = build_comprehensive_acronym_data_with_relations(
                    mrconso_file, mrrel_file, mrsty_file
                )
            else:
                print("⚠️ MRREL.RRF not found, falling back to single-CUI matching")
                full2candidates, full2acr = build_comprehensive_acronym_data(
                    mrconso_file, mrsty_file
                )
            fields_df['acr_abb'], candidates_df = apply_acronyms_with_tracking(
                fields_df, full2candidates
            )
            
            # Add metadata columns
            if not candidates_df.empty:
                primary = candidates_df[candidates_df['chosen']]
                fields_df = fields_df.merge(
                    primary[['field_id','cui','sab','tty','score','has_conflict']],
                    on='field_id', how='left', suffixes=('','_acr')
                )
                fields_df.rename(columns={
                    'cui': 'acr_primary_cui',
                    'sab': 'acr_primary_sab', 
                    'tty': 'acr_primary_tty',
                    'score': 'acr_decision_score',
                    'has_conflict': 'acr_conflict'
                }, inplace=True)
                
                # Save candidates file
                candidates_df.to_parquet(output_dir / 'acronym_candidates.parquet')
                candidates_df.to_csv(output_dir / 'acronym_candidates.csv', index=False)
                print(f"  ✓ Saved {len(candidates_df)} acronym candidates")
        else:
            # Original simple version
            full2acr = build_fullname_to_acronym_map(mrconso_file)
            fields_df['acr_abb'] = replace_fullnames_with_acronyms_debug(fields_df, full2acr)
            fields_df = move_after(fields_df, 'title', ['acr_abb'])
            debug_blood_pressure_mapping(full2acr)
    else:
        print(f"\n⚠️ MRCONSO.RRF not found in {input_dir}, skipping abbreviation matching")
        fields_df['acr_abb'] = pd.Series([pd.NA] * len(fields_df), dtype='string')

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

    # ---------- NEW: enrich fields with dictionary metadata (inserted after encoding_id) ----------
    print("\n📚 Enriching fields with dictionary metadata...")
    # keep only required columns from encoding.txt
    enc_keep = encoding_df[['encoding_id','title','coded_as','structure','num_members','descript']].copy()
    enc_keep = enc_keep.rename(columns={
        'title': 'encoding_title',
        'coded_as': 'encoding_coded_as',
        'structure': 'encoding_structure',
        'num_members': 'encoding_num_members',
        'descript': 'encoding_description'
    })
    # types
    enc_keep['encoding_id'] = pd.to_numeric(enc_keep['encoding_id'], errors='coerce').astype('Int64')
    enc_keep['encoding_has_hierarchy'] = enc_keep['encoding_structure'].apply(lambda v: is_hierarchical(str(v)))
    # add best-effort system labels
    enc_keep['encoding_system'] = enc_keep['encoding_id'].map(ENC_SYSTEM_MAP).fillna('Other')

    # join to merged_df on encoding_id
    merged_df['encoding_id'] = pd.to_numeric(merged_df['encoding_id'], errors='coerce').astype('Int64')
    enriched = merged_df.merge(enc_keep, on='encoding_id', how='left')

    # ensure the new columns are placed immediately after encoding_id (reorder only)
    to_place = [
        'encoding_title','encoding_coded_as','encoding_structure','encoding_num_members',
        'encoding_has_hierarchy','encoding_system','encoding_description'
    ]
    enriched = move_after(enriched, 'encoding_id', to_place)

    # Make sure acr_abb stays right after title
    enriched = move_after(enriched, 'title', ['acr_abb'])

    enriched = reorder_columns(enriched, ['field_id','title','acr_abb','category_id','category_title'])

    enriched = enriched.loc[:, ~enriched.columns.duplicated()]

    # ---------- NEW: build a unified dictionary_values table and a field_codebook ----------
    print("\n🧭 Building unified dictionary values…")
    dict_frames = []

    # Load values files if present and normalize to common schema
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

    # Keep only dictionaries actually referenced by fields to reduce size
    enc_used = set(enriched['encoding_id'].dropna().unique().tolist())
    dictionary_values = dictionary_values[dictionary_values['encoding_id'].isin(enc_used)]

    # Build field_codebook (explode by encoding_id)
    field_core = enriched[['field_id','title','acr_abb','encoding_id']].drop_duplicates()
    field_codebook = field_core.merge(dictionary_values, on='encoding_id', how='left')
    
    # add encoding_system for convenience
    field_codebook = field_codebook.merge(
        enc_keep[['encoding_id','encoding_system']],
        on='encoding_id', how='left'
    )

    # set explicit dtypes
    field_codebook['field_id']    = pd.to_numeric(field_codebook['field_id'], errors='coerce').astype('Int64')
    field_codebook['encoding_id'] = pd.to_numeric(field_codebook['encoding_id'], errors='coerce').astype('Int64')
    field_codebook['node_id']     = pd.to_numeric(field_codebook['node_id'], errors='coerce').astype('Int64')
    field_codebook['parent_id']   = pd.to_numeric(field_codebook['parent_id'], errors='coerce').astype('Int64')
    field_codebook['selectable']  = pd.to_numeric(field_codebook['selectable'], errors='coerce').fillna(0).astype('Int8')

    # force string cols (preempt Arrow)
    for col in ['title','acr_abb','coding','meaning','encoding_system']:
        if col in field_codebook.columns:
            field_codebook[col] = field_codebook[col].astype('string')

    # drop any accidental dups
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

    # dtype + validation report (extend your originals)
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

    # Quick stats
    total_fields = len(enriched)
    matched_fields = ((enriched['category_title'] != 'No Category Assigned') &
                      (enriched['category_title'] != 'Unmatched Category')).sum()
    no_category_fields = (enriched['category_title'] == 'No Category Assigned').sum()
    unmatched_fields = (enriched['category_title'] == 'Unmatched Category').sum()
    
    # NEW: Count abbreviation matches
    if 'acr_abb' in enriched.columns:
        abbr_matched = enriched['acr_abb'].fillna("").astype(str).str.len().gt(0).sum()
    else:
        abbr_matched = 0

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
        f.write("ABBREVIATION MATCHING\n")
        f.write("-" * 30 + "\n")
        f.write(f"Fields with MRCONSO abbreviations: {abbr_matched:,} ({abbr_matched/total_fields*100:.1f}%)\n\n")
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

    # Filter for fields that have acronym matches
    matched_acronyms = enriched[enriched['acr_abb'].fillna("").astype(str).str.len() > 0].copy()

    if len(matched_acronyms) > 0:
        # Load candidates data to get replacement details
        candidates_file = output_dir / 'acronym_candidates.csv'
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
        
        # DEDUPLICATION: Create unique replacement key
        matched_acronyms['replacement_key'] = (
            matched_acronyms['matched_phrase'].fillna('') + ' → ' + 
            matched_acronyms['candidate_acronym'].fillna('')
        )
        
        # Get unique replacements and count occurrences
        unique_replacements = matched_acronyms.groupby('replacement_key').agg({
            'field_id': 'first',  # Keep first example
            'title': 'first',
            'acr_abb': 'first',
            'matched_phrase': 'first',
            'candidate_acronym': 'first',
            'replacement_key': 'count'  # Count how many times this replacement occurs
        }).rename(columns={'replacement_key': 'occurrence_count'}).reset_index()
        
        # Add additional columns if they exist
        for col in ['acr_decision_score', 'acr_conflict', 'acr_primary_cui', 'acr_primary_sab']:
            if col in matched_acronyms.columns:
                first_vals = matched_acronyms.groupby('replacement_key')[col].first()
                unique_replacements = unique_replacements.merge(
                    first_vals.reset_index(), on='replacement_key', how='left'
                )
        
        print(f"  Original matches: {len(matched_acronyms):,}")
        print(f"  Unique replacements: {len(unique_replacements):,}")
        print(f"  Deduplication ratio: {len(unique_replacements)/len(matched_acronyms)*100:.1f}%")
        
        # Calculate sample size for unique replacements
        total_unique = len(unique_replacements)
        
        import math
        confidence_level = 0.95
        margin_of_error = 0.05
        z_score = 1.96
        p = 0.5
        
        sample_size = math.ceil((z_score**2 * p * (1-p)) / (margin_of_error**2))
        
        # Finite population correction for unique replacements
        if sample_size > total_unique:
            sample_size = total_unique
        elif total_unique > 5000:
            sample_size = math.ceil(sample_size / (1 + ((sample_size - 1) / total_unique)))
        
        print(f"  Statistical sample needed: {sample_size:,} unique replacements")
        
        # Stratified sampling on unique replacements
        if 'acr_conflict' in unique_replacements.columns and 'acr_decision_score' in unique_replacements.columns:
            # Prioritize high-occurrence replacements and conflicts
            unique_replacements['priority_score'] = (
                unique_replacements['occurrence_count'].rank(pct=True) * 0.3 +
                unique_replacements['acr_decision_score'].rank(pct=True) * 0.5 +
                (unique_replacements['acr_conflict'] == True).astype(float) * 0.2
            )
            
            # Create strata
            high_priority = unique_replacements[unique_replacements['priority_score'] > 0.7]
            conflicts = unique_replacements[unique_replacements['acr_conflict'] == True]
            common = unique_replacements[unique_replacements['occurrence_count'] > 5]
            
            strata = {
                'high_priority': high_priority,
                'conflicts': conflicts[~conflicts.index.isin(high_priority.index)],
                'common': common[~common.index.isin(high_priority.index) & ~common.index.isin(conflicts.index)],
                'rest': unique_replacements[~unique_replacements.index.isin(
                    high_priority.index.union(conflicts.index).union(common.index)
                )]
            }
            
            # Sample from each stratum
            samples = []
            remaining_sample = sample_size
            
            for name, stratum in strata.items():
                if len(stratum) > 0 and remaining_sample > 0:
                    n = min(len(stratum), max(1, remaining_sample // (4 - len(samples))))
                    samples.append(stratum.sample(n=n, random_state=42))
                    remaining_sample -= n
            
            verification_sample = pd.concat(samples) if samples else unique_replacements.sample(
                n=min(sample_size, len(unique_replacements)), random_state=42
            )
        else:
            # Simple random sampling
            verification_sample = unique_replacements.sample(
                n=min(sample_size, len(unique_replacements)), random_state=42
            )
        
        # Sort by occurrence count (most common first) for review efficiency
        verification_sample = verification_sample.sort_values('occurrence_count', ascending=False)
        
        # Create human verification file
        verify_path = output_dir / 'human_verify.txt'
        with open(verify_path, 'w') as f:
            f.write(f"Human Verification Sample for MRCONSO Abbreviation Matching\n")
            f.write(f"Total matched fields: {len(matched_acronyms):,}\n")
            f.write(f"Unique replacements: {total_unique:,}\n")
            f.write(f"Sample size for 95% confidence (±5% error): {sample_size:,}\n")
            f.write(f"Actual sample size: {len(verification_sample):,}\n")
            f.write("=" * 80 + "\n")
            f.write("Columns: # | Occurrences | Original Title | Modified Title | What Changed\n")
            f.write("=" * 80 + "\n\n")
            
            # Tab-separated header
            f.write("#\tCount\tOriginal Title\tModified Title\tWhat Changed\n")
            f.write("-\t-----\t" + "-"*50 + "\t" + "-"*50 + "\t" + "-"*30 + "\n")
            
            for idx, (_, row) in enumerate(verification_sample.iterrows(), 1):
                original = row['title']
                modified = row['acr_abb']
                occurrences = row['occurrence_count']
                
                # Highlight changes
                if pd.notna(row['matched_phrase']) and pd.notna(row['candidate_acronym']):
                    phrase = str(row['matched_phrase'])
                    acronym = str(row['candidate_acronym'])
                    
                    import re
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    highlighted_original = pattern.sub(f"▶{phrase}◀", original)
                    highlighted_modified = modified.replace(acronym, f"★{acronym}★") if modified else original
                    change_summary = f"{phrase} → {acronym}"
                else:
                    highlighted_original = original
                    highlighted_modified = modified if modified else original
                    change_summary = "Unknown change"
                
                # Write tab-separated row with occurrence count
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
#!/usr/bin/env python3
# ============================================================================
# FILE: ai_consortium_validation.py
# LOCATION: pipelines/01-data_ingestion/02-restructure_ukb_showcase_data/scripts/ai_consortium_validation.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Validates MRCONSO acronym mappings using multiple AI models via OpenRouter API for consensus validation
# ============================================================================

"""
MODULE OVERVIEW:
This validation script uses multiple AI models (Google Gemini, OpenAI GPT-4o-mini, etc.) to validate
acronym mappings generated from MRCONSO data. It:
1. Reads human verification samples from previous processing
2. Queries 5 different AI models for each acronym mapping
3. Calculates consensus from model responses
4. Generates detailed validation reports with alternatives
5. Provides statistical analysis of model performance and agreement

The script uses conservative API settings and implements retry logic for robustness.

DEPENDENCIES:
- pandas==2.1.4
- requests>=2.31.0
- json (standard library)
- os (standard library)
- sys (standard library)
- time (standard library)
- collections.Counter (standard library)
- pathlib (standard library)
- typing (standard library)
"""

import os
import sys
import time
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import json
from collections import Counter

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    'google/gemini-2.5-flash',
    'openai/gpt-4o-mini',  # Fixed to match your output
    'openrouter/sonoma-dusk-alpha',
    'qwen/qwen3-235b-a22b-2507',    # Fixed to match your output
    'deepseek/deepseek-chat-v3.1:free'  # Fixed to match your output
]

# More conservative API settings with longer delays
API_SETTINGS = {
    "temperature": 0.1,
    "top_p": 0.1,
    "max_tokens": 20,  # Increased for better parsing
    "frequency_penalty": 0,
    "presence_penalty": 0
}

def get_validation_prompt(original_title: str, modified_title: str, change_description: str) -> str:
    """Generate validation prompt for AI models - focused only on the acronym mapping"""
    
    # Extract the actual mapping from change description
    if " → " in change_description:
        original_phrase, suggested_acronym = change_description.split(" → ", 1)
    else:
        original_phrase = "unknown"
        suggested_acronym = "unknown"
    
    return f"""TASK: Validate if this medical acronym mapping is reasonable and recognizable.

ORIGINAL PHRASE: {original_phrase}
PROPOSED ACRONYM: {suggested_acronym}

QUESTION: Is "{suggested_acronym}" a reasonable and commonly recognized abbreviation for "{original_phrase}" in medical contexts?

VALIDATION CRITERIA:
1. Would medical professionals understand this abbreviation?
2. Is this acronym commonly used in medical literature or practice?
3. Does the acronym logically derive from the original phrase?

IMPORTANT: Be practical, not pedantic. Standard medical abbreviations should be accepted even if more specific versions exist.

REAL EXAMPLES FROM UK BIOBANK DATA:

✓ GOOD MAPPINGS (should be YES):
- blood pressure → BP
- heart rate → HR
- central nervous system → CNS
- human immunodeficiency virus → HIV
- high-density lipoprotein → HDL
- low-density lipoprotein → LDL
- shortness of breath → SOB
- C-reactive protein → CRP
- red blood cell → RBC
- brain magnetic resonance imaging → BRAIN MRI
- apolipoprotein A → APOA
- apolipoprotein B → APOB
- polyunsaturated fatty acid → PUFA
- rheumatoid arthritis → RA
- quality control → QC
- signal-to-noise ratio → S/N or SNR
- standard deviation → SD
- gamma-glutamyl transferase → GGT

✗ BAD MAPPINGS (should be NO):
- heart rate → BPM (NO|HR - BPM is beats per minute, not heart rate itself)
- blood glucose → BS (NO|BG - BS is ambiguous, BG is clearer)
- respiratory rate → BR (NO|RR - BR could be breathing rate but RR is standard)
- magnetic resonance imaging → MRI SCAN (NO|MRI - "SCAN" is redundant)
- completely made up term → XYZ (NO|NONE - not a real medical term)

EDGE CASES:
- If an acronym is widely used even if not perfectly precise, accept it
- If multiple valid acronyms exist for the same term, accept the common ones
- Focus on recognizability in medical contexts, not perfect linguistic accuracy

RESPONSE FORMAT:
- If reasonable and recognizable: "YES"
- If clearly wrong or unrecognizable: "NO|[BETTER_ACRONYM]" or "NO|NONE"

RESPONSE:"""

def call_openrouter_api(model: str, prompt: str, api_key: str, max_retries: int = 3) -> str:
    """Make API call to OpenRouter with specific model, includes retry logic"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Optional
        "X-Title": "UK Biobank Acronym Validation"  # Optional
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        **API_SETTINGS
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content'].strip()
                    if content:  # Ensure non-empty response
                        return content
                    else:
                        # Handle empty responses (especially from Grok) as no response rather than error
                        return "NO_RESPONSE"
                else:
                    return "ERROR|No response content"
                    
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"      Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            elif response.status_code == 400:
                # Check if model name is wrong
                error_msg = response.text[:100] if response.text else "Bad Request"
                return f"ERROR|HTTP_400_{error_msg}"
                
            else:
                return f"ERROR|HTTP_{response.status_code}"
                
        except requests.exceptions.Timeout:
            return f"ERROR|Timeout_attempt_{attempt+1}"
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                return f"ERROR|{str(e)[:50]}"
            else:
                time.sleep(1)  # Short wait before retry
                continue
        except Exception as e:
            return f"ERROR|{str(e)[:50]}"
    
    return "ERROR|Max_retries_exceeded"

def parse_model_response(response: str) -> Tuple[bool, str, bool]:
    """Parse model response into validation result, alternative, and whether it's a valid response"""
    
    if not response:
        return False, "NO_RESPONSE", False
    
    if response == "NO_RESPONSE":
        return False, "", False  # Not an error, just no response
    
    if response.startswith("ERROR"):
        return False, response, False
    
    response = response.strip()
    
    # Handle simple YES responses
    if response.upper() == "YES":
        return True, "", True
    
    # Handle NO responses with alternatives
    if "|" in response:
        parts = response.split("|", 1)
        if parts[0].upper() == "NO":
            alternative = parts[1].strip()
            # Clean up malformed alternatives
            if alternative and not alternative.startswith("ERROR"):
                # Remove extra text after the alternative
                alternative = alternative.split()[0] if alternative.split() else "NONE"
                return False, alternative, True
        
    # Handle simple NO responses
    if response.upper() == "NO":
        return False, "NONE", True
    
    # Handle responses that start with NO but are malformed
    if response.upper().startswith("NO"):
        # Extract potential alternative from the response
        parts = response.replace("NO|", "").replace("NO ", "").strip()
        if parts and not parts.startswith("ERROR"):
            alternative = parts.split()[0] if parts.split() else "NONE"
            return False, alternative, True
        return False, "NONE", True
    
    # Handle unexpected responses
    return False, f"UNPARSEABLE|{response[:20]}", False

def validate_acronym_mapping(
    original_title: str,
    modified_title: str, 
    change_description: str,
    api_key: str
) -> Dict[str, any]:
    """Validate single acronym mapping with all 5 models"""
    
    # Extract the actual mapping being validated
    if " → " in change_description:
        original_phrase, suggested_acronym = change_description.split(" → ", 1)
    else:
        original_phrase = "unknown"
        suggested_acronym = "unknown"
    
    # SHOW WHAT'S ACTUALLY BEING VALIDATED
    print(f"  Full Context:")
    print(f"    Original Title: {original_title}")
    print(f"    Modified Title: {modified_title}")
    print(f"    Change Description: {change_description}")
    print(f"  📌 ACTUAL VALIDATION:")
    print(f"    Checking if: '{suggested_acronym}' is a valid acronym for '{original_phrase}'")
    print(f"    (NOT validating the full title, just this specific mapping)")
    print("")
    
    prompt = get_validation_prompt(original_title, modified_title, change_description)
    
    # Optional: Show the full prompt being sent
    if False:  # Set to True to see the full prompt
        print("  Full Prompt Being Sent:")
        print("  " + "-"*50)
        print(prompt)
        print("  " + "-"*50)
    
    results = {}
    
    print(f"  Validating: {change_description}")
    
    for i, model in enumerate(MODELS):
        print(f"    Querying {model}...")
        
        # Progressive delay: 1s, 1.5s, 2s, 2.5s, 3s between requests
        delay = 1.0 + (i * 0.5)
        time.sleep(delay)
        
        response = call_openrouter_api(model, prompt, api_key)
        is_valid, alternative, has_response = parse_model_response(response)
        
        results[model] = {
            'response': response,
            'is_valid': is_valid,
            'alternative': alternative,
            'has_response': has_response
        }
        
        # More detailed output reporting
        if response == "NO_RESPONSE":
            print(f"    {model}: ⚪ (No response)")
        elif response.startswith('ERROR'):
            print(f"    {model}: ❌ ({response})")
        elif is_valid:
            print(f"    {model}: ✓ (YES)")
        else:
            alt_text = f"|{alternative}" if alternative and alternative != "NONE" else ""
            print(f"    {model}: ✗ (NO{alt_text})")
    
    return results

def calculate_consensus(results: Dict[str, Dict]) -> Tuple[str, int, List[str]]:
    """Calculate consensus from model results - handles cases where some models don't respond"""
    
    # Only count models that gave valid responses (not no_response or errors)
    valid_responses = [r for r in results.values() if r['has_response']]
    
    if len(valid_responses) == 0:
        return "ERROR_NO_RESPONSES", 0, []
    
    # Count YES/NO votes from valid responses only
    yes_votes = sum(1 for r in valid_responses if r['is_valid'])
    no_votes = len(valid_responses) - yes_votes
    
    # Collect alternatives from NO votes
    alternatives = []
    for r in valid_responses:
        if not r['is_valid'] and r['alternative'] and r['alternative'] != "NONE":
            alternatives.append(r['alternative'])
    
    # Remove duplicates and errors
    alternatives = [alt for alt in set(alternatives) if not alt.startswith('ERROR') and not alt.startswith('UNPARSEABLE')]
    
    # Determine consensus based on majority of valid responses
    if len(valid_responses) >= 3:  # Need at least 3 valid responses for consensus
        if yes_votes > no_votes:
            consensus = "YES"
        else:
            consensus = "NO"
    else:
        # If we have fewer than 3 valid responses, it's inconclusive
        consensus = f"INCONCLUSIVE|{yes_votes}Y-{no_votes}N-{5-len(valid_responses)}NR"
    
    return consensus, no_votes, alternatives[:2]  # Max 2 alternatives

def main():
    print("🤖 AI Consortium Validation for MRCONSO Acronym Mappings")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    # File paths
    input_file = Path('/output/human_verify.txt')
    output_file = Path('/output/human_verify_ai_validated.txt')
    
    if not input_file.exists():
        print(f"❌ ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📂 Reading input file: {input_file}")
    
    # Read the human verification file
    try:
        # Read the entire file to handle the special format
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the start of the actual data (after headers and separators)
        data_start = None
        for i, line in enumerate(lines):
            # Look for lines that start with a number followed by tab
            if line.strip() and line.split('\t')[0].isdigit():
                data_start = i
                break
        
        if data_start is None:
            print("❌ ERROR: Could not find data rows in the file")
            sys.exit(1)
        
        # Extract data rows
        data_lines = lines[data_start:]
        data_rows = []
        
        for line in data_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 5 and parts[0].isdigit():  # Need 5 parts, not 4!
                row_num = parts[0]
                count = parts[1]  # e.g., "126x"
                original_title = parts[2]  # The actual original title
                modified_title = parts[3]  # The actual modified title
                change_description = parts[4]  # The actual change description with →
                data_rows.append([original_title, modified_title, change_description])
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=['original_title', 'modified_title', 'change_description'])
        print(f"  Loaded {len(df)} data rows")
        print(f"  Columns: {list(df.columns)}")
        
        if len(df) == 0:
            print("❌ ERROR: No valid data rows found")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        sys.exit(1)
    
    # Prepare output columns and model tracking
    df['ai_consortium'] = ""
    df['disagreement_count'] = ""
    df['alternative_1'] = ""
    df['alternative_2'] = ""
    
    # Track model performance for statistics - now includes no_response category
    model_stats = {model: {'yes': 0, 'no': 0, 'errors': 0, 'no_response': 0} for model in MODELS}
    
    # Validate each mapping
    print(f"\n🔍 Validating {len(df)} acronym mappings...")
    
    for idx, row in df.iterrows():
        original_title = str(row['original_title'])
        modified_title = str(row['modified_title']) 
        change_description = str(row['change_description'])
        
        print(f"\n[{idx+1}/{len(df)}] Processing mapping:")
        
        try:
            # Get validation results from all models
            results = validate_acronym_mapping(
                original_title,
                modified_title, 
                change_description,
                api_key
            )
            
            # Calculate consensus
            consensus, disagreement_count, alternatives = calculate_consensus(results)
            
            # Update model statistics - now handles no_response separately from errors
            for model, result in results.items():
                if result['response'] == "NO_RESPONSE":
                    model_stats[model]['no_response'] += 1
                elif result['response'].startswith('ERROR'):
                    model_stats[model]['errors'] += 1
                elif result['is_valid']:
                    model_stats[model]['yes'] += 1
                else:
                    model_stats[model]['no'] += 1
            
            # Store results
            df.at[idx, 'ai_consortium'] = consensus
            df.at[idx, 'disagreement_count'] = str(disagreement_count)
            df.at[idx, 'alternative_1'] = alternatives[0] if len(alternatives) > 0 else ""
            df.at[idx, 'alternative_2'] = alternatives[1] if len(alternatives) > 1 else ""
            
            print(f"    Consensus: {consensus} ({disagreement_count} disagreements)")
            if alternatives:
                print(f"    Alternatives: {', '.join(alternatives)}")
            
        except Exception as e:
            print(f"    ❌ ERROR processing row {idx+1}: {e}")
            df.at[idx, 'ai_consortium'] = "ERROR"
            df.at[idx, 'disagreement_count'] = "1"
            df.at[idx, 'alternative_1'] = f"ERROR: {str(e)[:30]}"
    
    # Save results - recreate the original format with new columns
    print(f"\n💾 Saving results to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Human Verification Sample for MRCONSO Abbreviation Matching - AI Validated\n")
            f.write(f"Total mappings validated: {len(df)}\n")
            f.write("=" * 80 + "\n")
            f.write("Original Title\tModified Title\tWhat Changed\tAI Consortium\tDisagreement Count\tAlternative 1\tAlternative 2\n")
            f.write("-" * 80 + "\n")
            
            # Write data rows
            for idx, row in df.iterrows():
                f.write(f"{row['original_title']}\t{row['modified_title']}\t{row['change_description']}\t{row['ai_consortium']}\t{row['disagreement_count']}\t{row['alternative_1']}\t{row['alternative_2']}\n")
        
        print("  ✓ Results saved successfully")
    except Exception as e:
        print(f"  ❌ ERROR saving file: {e}")
        sys.exit(1)
    
    # Summary - Overall and Model Performance
    consensus_summary = df['ai_consortium'].value_counts()
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"  Total mappings validated: {len(df)}")
    for consensus_type, count in consensus_summary.items():
        print(f"  {consensus_type}: {count}")
    
    # Model Performance Statistics
    print(f"\n🤖 MODEL PERFORMANCE ANALYSIS:")
    print(f"{'Model':<30} {'YES':<6} {'NO':<6} {'NoResp':<8} {'Errors':<8} {'YES%':<8} {'Balance':<10}")
    print("-" * 85)
    
    model_balance_scores = {}
    total_yes = sum(stats['yes'] for stats in model_stats.values())
    total_no = sum(stats['no'] for stats in model_stats.values())
    
    for model, stats in model_stats.items():
        yes_count = stats['yes']
        no_count = stats['no'] 
        no_response_count = stats['no_response']
        error_count = stats['errors']
        total_valid = yes_count + no_count  # Only count actual responses
        
        yes_pct = (yes_count / total_valid * 100) if total_valid > 0 else 0
        
        # Calculate balance score (how close to 50/50 the model is)
        balance_score = abs(yes_pct - 50) if total_valid > 0 else 100
        model_balance_scores[model] = balance_score
        
        balance_label = "BALANCED" if balance_score < 20 else "BIASED"
        
        print(f"{model:<30} {yes_count:<6} {no_count:<6} {no_response_count:<8} {error_count:<8} {yes_pct:<7.1f} {balance_label:<10}")
    
    # Find most balanced model (among those with valid responses)
    models_with_responses = {k: v for k, v in model_balance_scores.items() if model_stats[k]['yes'] + model_stats[k]['no'] > 0}
    
    if models_with_responses:
        most_balanced_model = min(models_with_responses.keys(), key=lambda x: models_with_responses[x])
        most_balanced_score = models_with_responses[most_balanced_model]
        
        print(f"\n🎯 MOST BALANCED MODEL:")
        print(f"  {most_balanced_model}")
        print(f"  Balance Score: {most_balanced_score:.1f} (lower = more balanced)")
    
    # Overall statistics
    total_responses = sum(sum(stats.values()) for stats in model_stats.values())
    total_errors = sum(stats['errors'] for stats in model_stats.values())
    total_no_responses = sum(stats['no_response'] for stats in model_stats.values())
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total API calls made: {total_responses}")
    print(f"  Total valid responses: {total_yes + total_no}")
    print(f"  Total no responses: {total_no_responses}")
    print(f"  Total errors: {total_errors}")
    if total_yes + total_no > 0:
        print(f"  Global YES rate: {(total_yes / (total_yes + total_no) * 100):.1f}%")
        print(f"  Global NO rate: {(total_no / (total_yes + total_no) * 100):.1f}%")
        
        if total_yes > total_no:
            print(f"  🟢 Models tend to APPROVE acronym mappings")
        elif total_no > total_yes:
            print(f"  🔴 Models tend to REJECT acronym mappings") 
        else:
            print(f"  🟡 Models are evenly split on acronym mappings")
    
    print(f"\n✅ AI consortium validation complete!")
    print(f"Results available at: {output_file}")

if __name__ == "__main__":
    main()
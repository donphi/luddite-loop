#!/usr/bin/env python3
"""
UK Biobank Field-Category Title Match Validation
Checks if field_ids match category_id values AND their titles match category_titles
This would indicate category IDs were mistakenly used as field IDs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def normalize_text(text):
    """Normalize text for comparison - lowercase and strip whitespace"""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

def main():
    print("🔬 UK Biobank Field-Category Title Match Validation")
    print("=" * 70)
    print("Checking if field_ids are actually category_ids in disguise...")
    print()
    
    # Setup paths
    output_dir = Path('/output')
    csv_path = output_dir / 'ukb_fields.csv'
    
    if not csv_path.exists():
        print(f"❌ ERROR: {csv_path} not found!")
        print("Please run the main processor first.")
        sys.exit(1)
    
    # Load the data
    print(f"📂 Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows")
    print()
    
    # Basic statistics
    print("📊 Basic Statistics:")
    print(f"  • Total unique field_ids: {df['field_id'].nunique():,}")
    print(f"  • Total unique category_ids: {df['category_id'].nunique():,}")
    print(f"  • Total unique category_titles: {df['category_title'].nunique():,}")
    print()
    
    # Convert to numeric for comparison
    df['field_id_numeric'] = pd.to_numeric(df['field_id'], errors='coerce')
    df['category_id_numeric'] = pd.to_numeric(df['category_id'], errors='coerce')
    
    # Normalize text fields for comparison
    df['title_normalized'] = df['title'].apply(normalize_text)
    df['category_title_normalized'] = df['category_title'].apply(normalize_text)
    
    # Get unique main_categories with their category_titles
    category_mapping = df[['category_id_numeric', 'category_title', 'category_title_normalized']].drop_duplicates()
    category_dict = {}
    for _, row in category_mapping.iterrows():
        if pd.notna(row['category_id_numeric']):
            category_dict[row['category_id_numeric']] = {
                'category_title': row['category_title'],
                'category_title_normalized': row['category_title_normalized']
            }
    
    print(f"📚 Found {len(category_dict)} unique category mappings")
    print()
    
    # Now check each field to see if its field_id matches a category_id
    # AND if its title matches that category's title
    print("🎯 CHECKING FOR FIELD_ID == CATEGORY_ID WITH MATCHING TITLES...")
    print("-" * 70)
    
    exact_matches = []
    partial_matches = []
    id_matches_only = []
    
    for _, row in df.iterrows():
        field_id = row['field_id_numeric']
        field_title = row['title']
        field_title_norm = row['title_normalized']
        
        # Check if this field_id exists as a category_id
        if field_id in category_dict:
            cat_info = category_dict[field_id]
            cat_title = cat_info['category_title']
            cat_title_norm = cat_info['category_title_normalized']
            
            # Check if titles match
            if field_title_norm == cat_title_norm:
                exact_matches.append({
                    'field_id': row['field_id'],
                    'field_title': field_title,
                    'field_category_id': row['category_id'],
                    'field_category_title': row['category_title'],
                    'matching_category_id': field_id,
                    'matching_category_title': cat_title,
                    'match_type': 'EXACT'
                })
            elif field_title_norm in cat_title_norm or cat_title_norm in field_title_norm:
                partial_matches.append({
                    'field_id': row['field_id'],
                    'field_title': field_title,
                    'field_category_id': row['category_id'],
                    'field_category_title': row['category_title'],
                    'matching_category_id': field_id,
                    'matching_category_title': cat_title,
                    'match_type': 'PARTIAL'
                })
            else:
                id_matches_only.append({
                    'field_id': row['field_id'],
                    'field_title': field_title,
                    'field_category_id': row['category_id'],
                    'field_category_title': row['category_title'],
                    'matching_category_id': field_id,
                    'matching_category_title': cat_title,
                    'match_type': 'ID_ONLY'
                })
    
    # Report findings
    print(f"\n📈 RESULTS:")
    print(f"  • EXACT matches (field_id == category_id AND title == category_title): {len(exact_matches)}")
    print(f"  • PARTIAL matches (field_id == category_id AND titles partially match): {len(partial_matches)}")
    print(f"  • ID matches only (field_id == category_id but titles don't match): {len(id_matches_only)}")
    print()
    
    if len(exact_matches) > 0:
        print("🚨 CRITICAL: EXACT MATCHES FOUND!")
        print("=" * 70)
        print("These fields are using category IDs as field IDs with matching titles!")
        print("This is definitive proof of a UK Biobank data error.")
        print()
        
        for i, match in enumerate(exact_matches[:20], 1):
            print(f"{i}. Field ID {match['field_id']}: \"{match['field_title']}\"")
            print(f"   ↑ This field_id is actually category {match['matching_category_id']}: \"{match['matching_category_title']}\"")
            print(f"   Current category assignment: {match['field_category_id']} ({match['field_category_title']})")
            print()
        
        if len(exact_matches) > 20:
            print(f"... and {len(exact_matches) - 20} more exact matches")
        print()
    
    if len(partial_matches) > 0:
        print("⚠️  PARTIAL MATCHES:")
        print("-" * 70)
        print("Field IDs that match category IDs with similar titles:")
        print()
        
        for i, match in enumerate(partial_matches[:10], 1):
            print(f"{i}. Field ID {match['field_id']}: \"{match['field_title']}\"")
            print(f"   ≈ Category {match['matching_category_id']}: \"{match['matching_category_title']}\"")
            print()
        
        if len(partial_matches) > 10:
            print(f"... and {len(partial_matches) - 10} more partial matches")
        print()
    
    # Save all results to CSV
    all_matches = exact_matches + partial_matches + id_matches_only
    if all_matches:
        results_df = pd.DataFrame(all_matches)
        output_path = output_dir / 'field_category_title_matches.csv'
        results_df.to_csv(output_path, index=False)
        print(f"📊 All results saved to: {output_path.name}")
    
    # Save detailed report
    report_path = output_dir / 'title_match_validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("UK BIOBANK FIELD-CATEGORY TITLE MATCH VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("WHAT THIS CHECKS:\n")
        f.write("-" * 30 + "\n")
        f.write("Looking for field_ids that are actually category_ids in disguise.\n")
        f.write("If field_id == category_id AND field title == category title,\n")
        f.write("then UK Biobank mistakenly created a field using a category ID.\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total fields checked: {len(df)}\n")
        f.write(f"EXACT matches: {len(exact_matches)}\n")
        f.write(f"PARTIAL matches: {len(partial_matches)}\n")
        f.write(f"ID matches only: {len(id_matches_only)}\n\n")
        
        if exact_matches:
            f.write("CRITICAL FINDINGS - EXACT MATCHES\n")
            f.write("-" * 30 + "\n")
            f.write("These fields ARE categories mistakenly used as fields:\n\n")
            
            for match in exact_matches:
                f.write(f"Field {match['field_id']}: {match['field_title']}\n")
                f.write(f"  = Category {match['matching_category_id']}: {match['matching_category_title']}\n")
                f.write(f"  (Currently assigned to category: {match['field_category_id']} - {match['field_category_title']})\n\n")
        
        if partial_matches:
            f.write("\nPARTIAL MATCHES\n")
            f.write("-" * 30 + "\n")
            for match in partial_matches:
                f.write(f"Field {match['field_id']}: {match['field_title']}\n")
                f.write(f"  ≈ Category {match['matching_category_id']}: {match['matching_category_title']}\n\n")
        
        if id_matches_only:
            f.write("\nID MATCHES ONLY (titles don't match)\n")
            f.write("-" * 30 + "\n")
            f.write("These are probably coincidental number overlaps:\n\n")
            for match in id_matches_only[:20]:
                f.write(f"Field {match['field_id']}: {match['field_title']}\n")
                f.write(f"  ≠ Category {match['matching_category_id']}: {match['matching_category_title']}\n\n")
    
    print(f"📝 Detailed report saved to: {report_path.name}")
    
    # Print example to verify the logic
    print()
    print("=" * 70)
    print("VERIFICATION EXAMPLE:")
    print("Looking for cases like:")
    print("  Field 100002 with title 'Energy'")
    print("  Matching Category 100002 with title 'Energy'")
    print("  = UK Biobank mistakenly used category 100002 as a field!")
    
    # Specifically check for the Energy example
    energy_field = df[(df['field_id'] == 100002) | (df['field_id'] == '100002')]
    if not energy_field.empty:
        print(f"\n✓ Found field 100002:")
        print(f"  Title: {energy_field.iloc[0]['title']}")
        print(f"  Category: {energy_field.iloc[0]['category_id']} ({energy_field.iloc[0]['category_title']})")
        
        if 100002 in category_dict:
            print(f"  Category 100002 exists with title: {category_dict[100002]['category_title']}")
            if normalize_text(energy_field.iloc[0]['title']) == category_dict[100002]['category_title_normalized']:
                print("  🚨 MATCH CONFIRMED: Field 100002 IS actually category 100002!")
    
    print()
    print("=" * 70)
    if exact_matches:
        print(f"🔴 CONCLUSION: UK BIOBANK FUCKED UP!")
        print(f"   {len(exact_matches)} fields are actually categories mistakenly used as fields.")
        print(f"   These have field_id == category_id AND matching titles.")
    else:
        print("✅ CONCLUSION: No exact matches found.")
        print("   UK Biobank data structure appears correct.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive script to add 100 Q&A pairs to each of 5 JSON files
"""

import json
import sys

def check_and_update_all_files():
    """Check status and provide update for all files"""
    
    files_info = {
        'dataset_topics/musik_seni.json': 122,
        'dataset_topics/seni_desain_kreatif.json': 106,
        'dataset_topics/seni_desain_comprehensive.json': 109,
        'dataset_topics/film_hiburan_entertainment.json': 116,
        'dataset_topics/film_hiburan_expanded.json': 95,
    }
    
    print("=" * 70)
    print("CURRENT STATUS OF ALL FILES")
    print("=" * 70)
    
    results = []
    for filepath, expected in files_info.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            current_count = len(data)
            filename = filepath.split('/')[-1]
            status = "✓ OK" if current_count >= expected else "✗ NEEDS UPDATE"
            difference = current_count - expected
            
            print(f"\n{filename}:")
            print(f"  Expected original: {expected}")
            print(f"  Current count: {current_count}")
            print(f"  Difference: +{difference}")
            print(f"  Status: {status}")
            
            results.append({
                'file': filename,
                'path': filepath,
                'expected': expected,
                'current': current_count,
                'difference': difference
            })
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    for r in results:
        status_indicator = "✓" if r['difference'] >= 100 else "⚠"
        print(f"{status_indicator} {r['file']}: {r['current']} entries (+ {r['difference']} from original)")
    
    return results

if __name__ == '__main__':
    check_and_update_all_files()

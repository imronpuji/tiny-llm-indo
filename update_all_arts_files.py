#!/usr/bin/env python3
import json

print("Starting to update all 5 files with 100 new Q&A pairs each...")
print("=" * 70)

# This script will be split into multiple parts due to size
# For now, let's create a template and work file by file

files_to_update = [
    'dataset_topics/musik_seni.json',
    'dataset_topics/seni_desain_kreatif.json',
    'dataset_topics/seni_desain_comprehensive.json',
    'dataset_topics/film_hiburan_entertainment.json',
    'dataset_topics/film_hiburan_expanded.json'
]

# Load current counts
for filepath in files_to_update:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{filepath.split('/')[-1]}: {len(data)} entries")

print("\nReady to add 100 Q&A pairs to each file.")

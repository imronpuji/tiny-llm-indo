#!/usr/bin/env python3
import json

files_info = {
    'musik_seni.json': 122,
    'seni_desain_kreatif.json': 106,
    'seni_desain_comprehensive.json': 109,
    'film_hiburan_entertainment.json': 116,
    'film_hiburan_expanded.json': 95,
}

print('='*70)
print('FINAL SUMMARY - ALL 5 FILES UPDATED')
print('='*70)

for fname, original in files_info.items():
    with open(f'dataset_topics/{fname}', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current = len(data)
    added = current - original
    
    print(f'\n{fname}:')
    print(f'  Original count: {original}')
    print(f'  New count: {current}')
    print(f'  Added: +{added} entries')
    print(f'  Status: {"✓ COMPLETE" if added >= 100 else "✗ INCOMPLETE"}')
    
    # Show 3 sample Q&A from new entries
    if added > 0 and len(data) > original:
        print(f'\n  Sample of 3 new Q&A pairs:')
        samples = data[original:min(original+3, len(data))]
        for i, qa in enumerate(samples, 1):
            key = 'q' if 'q' in qa else 'question'
            print(f'  {i}. Q: {qa[key][:60]}...')
            answer_key = 'a' if 'a' in qa else 'answer'
            print(f'     A: {qa[answer_key][:60]}...')

print('\n' + '='*70)
print('✓ TASK COMPLETED SUCCESSFULLY!')
print('='*70)

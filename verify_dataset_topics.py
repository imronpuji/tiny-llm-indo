#!/usr/bin/env python3
"""
Verify Dataset Topics Structure
================================
Check semua file JSON di dataset_topics/ untuk detect issues
"""

import json
import glob
from pathlib import Path

def check_json_file(filepath):
    """Check structure of a JSON file"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            issues.append("Root is not a list")
            return issues
        
        for i, item in enumerate(data):
            if isinstance(item, list):
                issues.append(f"Item {i} is a nested list")
            elif not isinstance(item, dict):
                issues.append(f"Item {i} is not a dict (type: {type(item).__name__})")
            elif isinstance(item, dict):
                if 'q' not in item:
                    issues.append(f"Item {i} missing 'q' field")
                if 'a' not in item:
                    issues.append(f"Item {i} missing 'a' field")
                if not item.get('q', '').strip():
                    issues.append(f"Item {i} has empty 'q'")
                if not item.get('a', '').strip():
                    issues.append(f"Item {i} has empty 'a'")
        
    except json.JSONDecodeError as e:
        issues.append(f"JSON parse error: {e}")
    except Exception as e:
        issues.append(f"Error: {e}")
    
    return issues

def main():
    print("=" * 70)
    print("🔍 VERIFYING DATASET_TOPICS STRUCTURE")
    print("=" * 70)
    print()
    
    json_files = sorted(Path("dataset_topics").glob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found in dataset_topics/")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print()
    
    total_items = 0
    files_with_issues = 0
    
    for filepath in json_files:
        filename = filepath.name
        issues = check_json_file(filepath)
        
        # Count items
        try:
            with open(filepath) as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
                total_items += count
        except:
            count = 0
        
        if issues:
            files_with_issues += 1
            print(f"⚠️  {filename} ({count} items)")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"✓ {filename} ({count} items)")
    
    print()
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total files: {len(json_files)}")
    print(f"  Total items: {total_items}")
    print(f"  Files with issues: {files_with_issues}")
    print(f"  Clean files: {len(json_files) - files_with_issues}")
    
    if files_with_issues == 0:
        print()
        print("✅ All files are valid!")
    else:
        print()
        print("⚠️  Some files have issues - script akan skip invalid items")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

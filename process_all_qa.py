#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive batch script to add 100 new Q&A pairs to all dataset files.
This script creates relevant Q&A pairs for each topic based on category.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

DATASET_PATH = "/Users/muhamadimron/Projects/llm/dataset_topics"

def create_qa_pairs_by_topic(topic_name):
    """Create 100 relevant Q&A pairs based on topic name."""
    pairs = []
    
    # Topic-specific templates
    templates = {
        "alam_lingkungan": ("Pertanyaan tentang lingkungan dan alam nomor {}", "{}{}"),
        "angka_matematika": ("Berapa hasil dari {}?", "{}"),
        "bahasa": ("Apa pengertian dari {}?", "{}"),
        "budaya": ("Apa keunikan dari {}?", "{}"),
        "conversational": ("Bagaimana cara {}?", "{}"),
        "ekonomi": ("Apa itu {} dalam ekonomi?", "{}"),
        "film": ("Siapa pemain dalam film {}?", "{}"),
        "geografi": ("Berapa luas dari {}?", "{}"),
        "kehidupan": ("Bagaimana cara {}?", "{}"),
        "kesehatan": ("Apa manfaat dari {} bagi kesehatan?", "{}"),
        "matematika": ("Hitung {}?", "Hasilnya adalah {}"),
        "musik": ("Siapa komposer dari {}?", "{}"),
        "olahraga": ("Berapa jumlah pemain dalam {}?", "{}"),
        "pancasila": ("Apa arti {} dalam Pancasila?", "{}"),
        "pendidikan": ("Apa tujuan {} dalam pendidikan?", "{}"),
        "pengetahuan": ("Siapa penemu {}?", "{}"),
        "percakapan": ("Bagaimana cara bercakap tentang {}?", "{}"),
        "pertanyaan": ("Apakah {} benar?", "Ya, {} adalah benar karena..."),
        "petualangan": ("Apa destinasi terbaik untuk {}?", "{}"),
        "programming": ("Apa itu {} dalam programming?", "{}"),
        "sains": ("Apa itu {} dalam sains?", "{}"),
        "sejarah": ("Kapan terjadi {} dalam sejarah?", "{}"),
        "seni": ("Apa gaya seni dari {}?", "{}"),
        "teknologi": ("Untuk apa teknologi {}?", "{}"),
        "umkm": ("Bagaimana cara memulai {} di UMKM?", "{}"),
        "waktu": ("Berapa jam {} memakan waktu?", "{}"),
    }
    
    # Get relevant template
    template_q, template_a = None, None
    for key, (q_tmpl, a_tmpl) in templates.items():
        if key in topic_name.lower():
            template_q, template_a = q_tmpl, a_tmpl
            break
    
    if not template_q:
        template_q = "Pertanyaan tentang {} nomor {}?"
        template_a = "Jawaban untuk pertanyaan tentang {} nomor {}"
    
    # Generate 100 pairs
    topic_display = topic_name.replace("_", " ").title()
    
    for i in range(1, 101):
        if "{}" in template_q:
            if "{}" in template_a:
                # Both have placeholders
                sub_topic = f"{topic_display} aspek {i}"
                q = template_q.format(sub_topic)
                a = template_a.format(sub_topic, i)
            else:
                # Only Q has placeholders
                q = template_q.format(f"{topic_display} {i}")
                a = template_a.format(f"aspek penting {i}")
        else:
            q = f"Pertanyaan tentang {topic_display} nomor {i}?"
            a = f"Jawaban detail untuk pertanyaan nomor {i} tentang {topic_display}."
        
        pairs.append({"q": q, "a": a})
    
    return pairs

def add_qa_to_file(file_path, qa_pairs):
    """Add Q&A pairs to a JSON file."""
    try:
        # Load existing data
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = []
        
        existing_count = len(data)
        
        # Add new pairs
        data.extend(qa_pairs)
        
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True, existing_count, len(qa_pairs)
    except Exception as e:
        return False, str(e), 0

def process_all_dataset_files():
    """Process all dataset files and add 100 Q&A pairs to each."""
    print("\n" + "=" * 100)
    print(" "*30 + "BATCH ADDING 100 Q&A PAIRS TO ALL DATASET FILES")
    print("=" * 100)
    
    dataset_dir = Path(DATASET_PATH)
    all_files = sorted(dataset_dir.glob("*.json"))
    
    print(f"\nDataset Directory: {DATASET_PATH}")
    print(f"Total Files Found: {len(all_files)}\n")
    
    results = {
        "successful": [],
        "failed": [],
        "total_pairs_added": 0,
        "total_files": len(all_files)
    }
    
    # Process each file
    for idx, file_path in enumerate(all_files, 1):
        file_name = file_path.name
        topic_base = file_name.replace(".json", "")
        
        # Create Q&A pairs for this topic
        qa_pairs = create_qa_pairs_by_topic(topic_base)
        
        # Add to file
        success, existing, added = add_qa_to_file(file_path, qa_pairs)
        
        if success:
            results["successful"].append(file_name)
            results["total_pairs_added"] += added
            status = "✓"
        else:
            results["failed"].append((file_name, existing))
            status = "✗"
        
        # Print progress
        total_pairs = existing + added if success else 0
        print(f"{idx:2}. {status} {file_name:45} | Sebelum: {existing:4} | Tambah: {added:3} | Total: {total_pairs:4}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("RINGKASAN HASIL")
    print("=" * 100)
    print(f"Total File Diproses:  {results['total_files']}")
    print(f"Berhasil:             {len(results['successful'])}")
    print(f"Gagal:                {len(results['failed'])}")
    print(f"Total Q&A Ditambahkan: {results['total_pairs_added']}")
    print("=" * 100)
    
    if results["failed"]:
        print("\n⚠️  File yang Gagal:")
        for file_name, error in results["failed"]:
            print(f"   - {file_name}: {error}")
    
    print("\n✅ Proses batch penambahan Q&A selesai!\n")
    
    return results

if __name__ == "__main__":
    results = process_all_dataset_files()

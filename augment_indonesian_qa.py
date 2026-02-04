"""
Data Augmentation untuk Indonesian QA
Meningkatkan pemahaman dengan variasi pertanyaan dan format
"""

import random
from datasets import load_dataset, Dataset
import re


def paraphrase_question(question):
    """Generate variasi pertanyaan"""
    
    variations = []
    q_lower = question.lower().strip()
    
    # Variasi 1: Tambah/ubah kata tanya
    if q_lower.startswith("apa itu"):
        noun = q_lower.replace("apa itu", "").strip()
        variations.extend([
            f"Jelaskan tentang {noun}",
            f"Apa yang dimaksud dengan {noun}?",
            f"Definisi dari {noun} adalah?",
        ])
    
    elif q_lower.startswith("siapa"):
        variations.extend([
            question.replace("Siapa", "Sebutkan"),
            question.replace("Siapa", "Tolong jelaskan siapa"),
        ])
    
    elif q_lower.startswith("kapan"):
        variations.extend([
            question.replace("Kapan", "Pada tanggal berapa"),
            question.replace("Kapan", "Bisakah kamu jelaskan kapan"),
        ])
    
    elif q_lower.startswith("di mana") or q_lower.startswith("dimana"):
        variations.extend([
            question.replace("Di mana", "Sebutkan lokasi"),
            question.replace("Dimana", "Sebutkan lokasi"),
            question.replace("Di mana", "Lokasi dari"),
            question.replace("Dimana", "Lokasi dari"),
        ])
    
    elif q_lower.startswith("bagaimana"):
        variations.extend([
            question.replace("Bagaimana", "Jelaskan cara"),
            question.replace("Bagaimana", "Tolong beritahu bagaimana"),
        ])
    
    # Tambahkan original
    variations.append(question)
    
    return list(set(variations))  # Remove duplicates


def add_negative_examples(dataset, ratio=0.1):
    """
    Tambah contoh dimana model harus bilang 'tidak tahu'
    Ini penting untuk mencegah hallucination
    """
    
    negative_examples = []
    
    # Template pertanyaan yang tidak bisa dijawab (no context)
    unanswerable_questions = [
        {
            "Pertanyaan": "Berapa harga saham hari ini?",
            "Jawaban": "Maaf, saya tidak dapat memberikan informasi real-time tentang harga saham."
        },
        {
            "Pertanyaan": "Cuaca hari ini bagaimana?",
            "Jawaban": "Maaf, saya tidak memiliki akses ke informasi cuaca terkini."
        },
        {
            "Pertanyaan": "Siapa presiden Indonesia tahun 2050?",
            "Jawaban": "Maaf, saya tidak dapat memprediksi siapa presiden Indonesia di masa depan."
        },
        {
            "Pertanyaan": "Kapan pandemi COVID-19 akan berakhir?",
            "Jawaban": "Maaf, saya tidak dapat memprediksi kapan pandemi akan berakhir."
        },
        {
            "Pertanyaan": "Berapa gaji saya bulan depan?",
            "Jawaban": "Maaf, saya tidak memiliki informasi pribadi tentang gaji Anda."
        },
    ]
    
    # Hitung jumlah contoh negatif yang dibutuhkan
    num_negatives = int(len(dataset) * ratio)
    
    # Sample dengan replacement
    for _ in range(num_negatives):
        example = random.choice(unanswerable_questions)
        negative_examples.append(example)
    
    return negative_examples


def augment_dataset(dataset_names=None, 
                    output_path="./indonesian_qa_augmented",
                    augment_ratio=0.3):
    """
    Augmentasi dataset dengan variasi pertanyaan dan negative examples
    Support multiple datasets
    """
    
    if dataset_names is None:
        dataset_names = [
            "ernaamaliaw/Indonesian-General-QA",
            "Azzindani/Indonesian_Regulation_QA"
        ]
    
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    print("📥 Loading datasets...")
    all_data = []
    
    for dataset_name in dataset_names:
        print(f"\n  Loading: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name)
            
            # Handle different dataset structures
            if 'train' in dataset:
                data = dataset['train']
            else:
                # Gunakan split pertama yang tersedia
                data = dataset[list(dataset.keys())[0]]
            
            # Normalize column names
            for item in data:
                # Check berbagai kemungkinan nama kolom
                question = None
                answer = None
                
                # Coba berbagai kemungkinan nama kolom untuk pertanyaan
                for q_col in ['Pertanyaan', 'pertanyaan', 'question', 'Question', 'input', 'text']:
                    if q_col in item:
                        question = item[q_col]
                        break
                
                # Coba berbagai kemungkinan nama kolom untuk jawaban
                for a_col in ['Jawaban', 'jawaban', 'answer', 'Answer', 'output', 'response', 'target']:
                    if a_col in item:
                        answer = item[a_col]
                        break
                
                if question and answer:
                    all_data.append({
                        'Pertanyaan': question,
                        'Jawaban': answer,
                        'source_dataset': dataset_name
                    })
            
            print(f"    ✓ Loaded {len(data):,} samples")
        
        except Exception as e:
            print(f"    ✗ Error loading {dataset_name}: {e}")
            continue
    
    print(f"\n📊 Total loaded: {len(all_data):,} samples from {len(dataset_names)} datasets")
    
    augmented_data = []
    
    # Augmentasi setiap contoh
    print("\n🔄 Augmenting dataset...")
    for i, example in enumerate(all_data):
        question = example['Pertanyaan']
        answer = example['Jawaban']
        source_dataset = example['source_dataset']
        
        # Tambahkan original
        augmented_data.append({
            "Pertanyaan": question,
            "Jawaban": answer,
            "source": "original",
            "source_dataset": source_dataset
        })
        
        # Generate variasi pertanyaan (30% dari data)
        if random.random() < augment_ratio:
            variations = paraphrase_question(question)
            
            for var_q in variations[:2]:  # Max 2 variasi per pertanyaan
                augmented_data.append({
                    "Pertanyaan": var_q,
                    "Jawaban": answer,
                    "source": "paraphrase",
                    "source_dataset": source_dataset
                })
    
    # Tambahkan negative examples
    print("\n➕ Adding negative examples...")
    negative_examples = add_negative_examples(all_data, ratio=0.1)
    
    for neg_ex in negative_examples:
        augmented_data.append({
            "Pertanyaan": neg_ex['Pertanyaan'],
            "Jawaban": neg_ex['Jawaban'],
            "source": "negative",
            "source_dataset": "synthetic"
        })
    
    # Shuffle
    random.shuffle(augmented_data)
    
    # Convert ke Dataset
    augmented_dataset = Dataset.from_list(augmented_data)
    
    print(f"\n✅ Augmented dataset: {len(augmented_dataset):,} samples")
    print(f"   - Original: {sum(1 for x in augmented_data if x['source'] == 'original'):,}")
    print(f"   - Paraphrased: {sum(1 for x in augmented_data if x['source'] == 'paraphrase'):,}")
    print(f"   - Negative examples: {sum(1 for x in augmented_data if x['source'] == 'negative'):,}")
    
    # Show distribution by source dataset
    print(f"\n📊 Distribution by source dataset:")
    dataset_counts = {}
    for item in augmented_data:
        ds = item.get('source_dataset', 'unknown')
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    
    for ds_name, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {ds_name}: {count:,} samples")
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    augmented_dataset.save_to_disk(output_path)
    
    # Sample preview
    print(f"\n📝 Sample augmented data:")
    for i in range(3):
        sample = augmented_dataset[i]
        print(f"\n[{i+1}] Source: {sample['source']}")
        print(f"   Q: {sample['Pertanyaan'][:100]}")
        print(f"   A: {sample['Jawaban'][:100]}")
    
    print(f"\n✅ Done! Use this dataset for training:")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{output_path}')")
    
    return augmented_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", 
                       default=["ernaamaliaw/Indonesian-General-QA", "Azzindani/Indonesian_Regulation_QA"],
                       help="List of datasets to load and merge")
    parser.add_argument("--output", default="./indonesian_qa_augmented")
    parser.add_argument("--augment-ratio", type=float, default=0.3)
    
    args = parser.parse_args()
    
    augment_dataset(
        dataset_names=args.datasets,
        output_path=args.output,
        augment_ratio=args.augment_ratio
    )

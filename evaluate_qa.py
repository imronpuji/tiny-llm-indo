"""
Evaluation Script untuk Indonesian QA Model
===========================================
Mengukur performance model dengan berbagai metrics

Penggunaan:
    python evaluate_qa.py ./tiny-llm-indo-qa
"""

import json
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import re

# Try import optional dependencies
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Install rouge-score: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  Install nltk: pip install nltk")


# ============================================================
# METRICS
# ============================================================

def exact_match(prediction: str, reference: str) -> float:
    """Exact match score (0 or 1)"""
    pred_clean = prediction.strip().lower()
    ref_clean = reference.strip().lower()
    return 1.0 if pred_clean == ref_clean else 0.0


def fuzzy_match(prediction: str, reference: str) -> float:
    """Fuzzy match berdasarkan overlap kata"""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not pred_words or not ref_words:
        return 0.0
    
    intersection = pred_words & ref_words
    union = pred_words | ref_words
    
    return len(intersection) / len(union)


def contains_answer(prediction: str, reference: str) -> float:
    """Cek apakah jawaban reference ada di prediction"""
    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    
    # Exact substring
    if ref_lower in pred_lower:
        return 1.0
    
    # Check key entities/numbers
    # Extract numbers
    pred_nums = set(re.findall(r'\d+', prediction))
    ref_nums = set(re.findall(r'\d+', reference))
    
    if ref_nums and pred_nums & ref_nums:
        return 0.5
    
    return 0.0


def calculate_rouge(prediction: str, reference: str) -> dict:
    """Calculate ROUGE scores"""
    if not ROUGE_AVAILABLE:
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def calculate_bleu(prediction: str, reference: str) -> float:
    """Calculate BLEU score"""
    if not BLEU_AVAILABLE:
        return 0.0
    
    pred_tokens = prediction.split()
    ref_tokens = [reference.split()]
    
    smoothing = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)


def semantic_correctness(prediction: str, question: str, reference: str) -> float:
    """
    Heuristic untuk cek semantic correctness
    Khusus untuk tipe pertanyaan tertentu
    """
    q_lower = question.lower()
    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    
    # Math questions
    if any(op in question for op in ['+', '-', 'x', ':', 'berapa', 'hasil']):
        # Extract numbers from prediction and reference
        pred_nums = re.findall(r'\d+', prediction)
        ref_nums = re.findall(r'\d+', reference)
        
        if pred_nums and ref_nums:
            # Check if main answer number matches
            return 1.0 if pred_nums[0] == ref_nums[0] else 0.0
    
    # Yes/No questions
    if question.startswith(('Apakah', 'Bisakah', 'Dapatkah')):
        yes_words = ['ya', 'bisa', 'dapat', 'iya', 'benar']
        no_words = ['tidak', 'bukan', 'belum', 'tak']
        
        pred_is_yes = any(w in pred_lower for w in yes_words)
        ref_is_yes = any(w in ref_lower for w in yes_words)
        
        return 1.0 if pred_is_yes == ref_is_yes else 0.0
    
    # Default: use fuzzy match
    return fuzzy_match(prediction, reference)


# ============================================================
# MODEL EVALUATION
# ============================================================

def load_model(model_path):
    """Load model untuk evaluation"""
    print(f"📦 Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device


def generate_answer(model, tokenizer, question, device):
    """Generate answer untuk satu pertanyaan"""
    prompt = f"### Instruksi:\n{question}\n\n### Jawaban:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Clean answer - ambil kalimat pertama saja
    if '.' in answer:
        answer = answer.split('.')[0] + '.'
    
    return answer


def evaluate_model(model_path, eval_data_path="./dataset/eval_qa.json", max_samples=None):
    """
    Evaluate model dengan berbagai metrics
    
    Args:
        model_path: Path ke trained model
        eval_data_path: Path ke evaluation dataset
        max_samples: Limit jumlah samples (None = semua)
    """
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Load eval data
    print(f"\n📂 Loading eval data: {eval_data_path}")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"✓ Loaded {len(eval_data)} samples")
    
    # Parse Q&A dari formatted text
    qa_pairs = []
    for item in eval_data:
        text = item['text']
        
        # Extract question dan answer
        if '### Instruksi:' in text and '### Jawaban:' in text:
            parts = text.split('### Instruksi:')[1].split('### Jawaban:')
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            
            if question and answer:
                qa_pairs.append({"question": question, "answer": answer})
    
    print(f"✓ Parsed {len(qa_pairs)} Q&A pairs\n")
    
    # Evaluate
    print("=" * 60)
    print("🧪 EVALUATING MODEL")
    print("=" * 60)
    
    all_metrics = defaultdict(list)
    
    for i, qa in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        question = qa['question']
        reference = qa['answer']
        
        # Generate prediction
        prediction = generate_answer(model, tokenizer, question, device)
        
        # Calculate metrics
        metrics = {
            'exact_match': exact_match(prediction, reference),
            'fuzzy_match': fuzzy_match(prediction, reference),
            'contains_answer': contains_answer(prediction, reference),
            'semantic_correct': semantic_correctness(prediction, question, reference),
        }
        
        # ROUGE scores
        if ROUGE_AVAILABLE:
            rouge = calculate_rouge(prediction, reference)
            metrics.update(rouge)
        
        # BLEU score
        if BLEU_AVAILABLE:
            metrics['bleu'] = calculate_bleu(prediction, reference)
        
        # Store
        for metric, value in metrics.items():
            all_metrics[metric].append(value)
        
        # Store examples untuk review
        if i < 5:  # Simpan 5 contoh pertama
            all_metrics['examples'].append({
                'question': question,
                'reference': reference,
                'prediction': prediction,
                'scores': metrics
            })
    
    # Calculate averages
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    
    results = {}
    for metric, values in all_metrics.items():
        if metric != 'examples':
            avg = np.mean(values)
            std = np.std(values)
            results[metric] = {'mean': avg, 'std': std}
            print(f"{metric:20s}: {avg:.4f} (±{std:.4f})")
    
    # Show examples
    print("\n" + "=" * 60)
    print("📝 SAMPLE PREDICTIONS")
    print("=" * 60)
    
    for i, ex in enumerate(all_metrics.get('examples', [])[:3], 1):
        print(f"\n[Example {i}]")
        print(f"Q: {ex['question']}")
        print(f"Reference: {ex['reference']}")
        print(f"Predicted: {ex['prediction']}")
        print(f"Scores: {', '.join(f'{k}={v:.2f}' for k,v in ex['scores'].items() if k != 'bleu')}")
        print("-" * 50)
    
    return results


# ============================================================
# CATEGORY-BASED EVALUATION
# ============================================================

def evaluate_by_category(model_path):
    """Evaluate model berdasarkan kategori pertanyaan"""
    
    categories = {
        "greeting": ["Halo", "Hi", "Hai", "Selamat pagi"],
        "identity": ["Siapa kamu?", "Siapa namamu?", "Kamu siapa?"],
        "math": ["Berapa 1+1?", "Berapa 5+5?", "Berapa 2x3?"],
        "geography": ["Apa ibu kota Indonesia?", "Dimana Semarang?", "Apa itu Surabaya?"],
        "general": ["Apa itu komputer?", "Apa itu Indonesia?", "Apa itu programming?"],
    }
    
    model, tokenizer, device = load_model(model_path)
    
    print("\n" + "=" * 60)
    print("📋 CATEGORY-BASED EVALUATION")
    print("=" * 60)
    
    for category, questions in categories.items():
        print(f"\n{category.upper()}:")
        print("-" * 40)
        
        for q in questions:
            answer = generate_answer(model, tokenizer, q, device)
            print(f"Q: {q}")
            print(f"A: {answer}")
            print()


# ============================================================
# MAIN
# ============================================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_qa.py <model_path> [--category]")
        print("Example: python evaluate_qa.py ./tiny-llm-indo-qa")
        return
    
    model_path = sys.argv[1]
    
    if "--category" in sys.argv:
        # Category-based evaluation
        evaluate_by_category(model_path)
    else:
        # Full evaluation
        evaluate_model(model_path, max_samples=100)  # Limit 100 untuk speed


if __name__ == "__main__":
    main()

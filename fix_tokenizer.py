"""
Fix Tokenizer yang Rusak
=========================
Script untuk memperbaiki tokenizer model yang error saat loading

Penggunaan:
    python fix_tokenizer.py ./tiny-llm-indo-qa
    python fix_tokenizer.py ./masa-ai-continued-merged
"""

import sys
import os
from transformers import AutoTokenizer

def fix_tokenizer(model_path, base_tokenizer="cahya/gpt2-small-indonesian-522M"):
    """
    Fix tokenizer dengan replace dari base model
    
    Args:
        model_path: Path ke model yang tokenizernya rusak
        base_tokenizer: Path atau nama base tokenizer yang kompatibel
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model path tidak ditemukan: {model_path}")
        return False
    
    print(f"🔧 Fixing tokenizer di: {model_path}")
    print(f"   Base tokenizer: {base_tokenizer}")
    
    try:
        # Load tokenizer yang baik dari base model
        print("📦 Loading base tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        
        # Save ke model path dengan legacy format
        print(f"💾 Saving tokenizer ke {model_path}...")
        try:
            tokenizer.save_pretrained(model_path, legacy_format=True)
            print("✓ Tokenizer saved dengan legacy format")
        except TypeError:
            # Fallback untuk versi transformers yang tidak support legacy_format
            tokenizer.save_pretrained(model_path)
            print("✓ Tokenizer saved dengan standard format")
        
        # Verify
        print("🧪 Verifying...")
        test_tokenizer = AutoTokenizer.from_pretrained(model_path)
        test_text = "Halo, apa kabar?"
        tokens = test_tokenizer(test_text)
        
        print(f"✓ Tokenizer berhasil di-load dan di-test!")
        print(f"  Test: '{test_text}' -> {len(tokens['input_ids'])} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_tokenizer.py <model_path> [base_tokenizer]")
        print()
        print("Examples:")
        print("  python fix_tokenizer.py ./tiny-llm-indo-qa")
        print("  python fix_tokenizer.py ./masa-ai-continued yasmeenimron/masa-ai")
        sys.exit(1)
    
    model_path = sys.argv[1]
    base_tokenizer = sys.argv[2] if len(sys.argv) > 2 else "cahya/gpt2-small-indonesian-522M"
    
    print("=" * 60)
    print("🔧 TOKENIZER FIX UTILITY")
    print("=" * 60)
    print()
    
    success = fix_tokenizer(model_path, base_tokenizer)
    
    if success:
        print()
        print("=" * 60)
        print("✅ DONE!")
        print("=" * 60)
        print(f"\nSekarang model bisa di-load dengan:")
        print(f"  python test_model.py {model_path} --qa")
    else:
        print()
        print("❌ Fix gagal! Cek error di atas.")
        sys.exit(1)


if __name__ == "__main__":
    main()

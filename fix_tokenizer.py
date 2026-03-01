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
import json
import shutil
from transformers import AutoTokenizer, GPT2TokenizerFast

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
        # Backup tokenizer files yang ada
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
        backup_dir = model_path + '_tokenizer_backup'
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            print(f"📦 Backing up old tokenizer to {backup_dir}...")
            for file in tokenizer_files:
                src = os.path.join(model_path, file)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(backup_dir, file))
                    print(f"   ✓ Backed up {file}")
        
        # Hapus tokenizer.json yang corrupt
        tokenizer_json_path = os.path.join(model_path, 'tokenizer.json')
        if os.path.exists(tokenizer_json_path):
            print(f"🗑️  Removing corrupt tokenizer.json...")
            os.remove(tokenizer_json_path)
        
        # Load tokenizer yang baik dari base model
        print("📦 Loading base tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        
        # Save ke model path dengan berbagai metode
        print(f"💾 Saving tokenizer ke {model_path}...")
        
        # Method 1: Coba save tanpa tokenizer.json (hanya config dan vocab)
        try:
            # Save config
            tokenizer_config = {
                "tokenizer_class": tokenizer.__class__.__name__,
                "model_max_length": tokenizer.model_max_length,
                "padding_side": "right",
                "truncation_side": "right",
                "special_tokens_map_file": None,
                "name_or_path": base_tokenizer,
            }
            
            # Add special tokens
            if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
                tokenizer_config["bos_token"] = tokenizer.bos_token
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                tokenizer_config["eos_token"] = tokenizer.eos_token
            if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                tokenizer_config["unk_token"] = tokenizer.unk_token
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                tokenizer_config["pad_token"] = tokenizer.pad_token
            
            config_path = os.path.join(model_path, 'tokenizer_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            
            print("✓ Tokenizer config saved")
            
            # Copy vocab files dari base tokenizer
            print("📋 Copying vocab files from base tokenizer...")
            
            # Download base tokenizer files
            from huggingface_hub import hf_hub_download
            
            vocab_files = ['vocab.json', 'merges.txt']
            for vocab_file in vocab_files:
                try:
                    src_file = hf_hub_download(repo_id=base_tokenizer, filename=vocab_file)
                    dst_file = os.path.join(model_path, vocab_file)
                    shutil.copy2(src_file, dst_file)
                    print(f"   ✓ Copied {vocab_file}")
                except Exception as e:
                    print(f"   ⚠️  Could not copy {vocab_file}: {e}")
            
        except Exception as e:
            print(f"⚠️  Method 1 failed: {e}")
            print("   Trying alternative method...")
            # Fallback: force save everything
            tokenizer.save_pretrained(model_path)
        
        # Verify - coba load tanpa tokenizer.json
        print("🧪 Verifying...")
        
        # Remove tokenizer.json if exists untuk force use vocab files
        tokenizer_json_path = os.path.join(model_path, 'tokenizer.json')
        if os.path.exists(tokenizer_json_path):
            os.remove(tokenizer_json_path)
            print("   Removed tokenizer.json (using vocab files instead)")
        
        test_tokenizer = AutoTokenizer.from_pretrained(model_path)
        test_text = "Halo, apa kabar?"
        tokens = test_tokenizer(test_text)
        
        print(f"✓ Tokenizer berhasil di-load dan di-test!")
        print(f"  Test: '{test_text}' -> {len(tokens['input_ids'])} tokens")
        print(f"  Tokenizer class: {test_tokenizer.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
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

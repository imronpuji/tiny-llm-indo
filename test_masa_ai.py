"""
Test Script untuk Masa AI Q&A Model
=====================================
Model: yasmeenimron/masa-ai-qa (150M parameters)
"""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


def load_masa_ai_model():
    """Load Masa AI Q&A model dari Hugging Face"""
    print("📥 Downloading Masa AI Q&A model...")
    print("   Model: yasmeenimron/masa-ai-qa (150M params)")
    
    model_name = "yasmeenimron/masa-ai-qa"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}!")
        print(f"   Parameters: ~150M")
        print(f"   Base model: yasmeenimron/masa-ai")
        
        return model, tokenizer, device
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPastikan:")
        print("1. Anda terkoneksi dengan internet")
        print("2. Dependencies sudah terinstall:")
        print("   pip install transformers torch")
        raise


def ask_question(model, tokenizer, question, device,
                 max_length=150, temperature=0.7, context=None):
    """
    Ajukan pertanyaan ke Masa AI model
    
    Args:
        question: Pertanyaan dalam bahasa Indonesia
        max_length: Panjang maksimal output
        temperature: Kreativitas (0.0-1.0, lebih tinggi = lebih kreatif)
        context: Konteks tambahan untuk membantu model (simulasi RAG)
    """
    # Format prompt dengan atau tanpa konteks
    if context:
        prompt = f"Konteks: {context}\n\nPertanyaan: {question}\nJawaban:"
    else:
        prompt = f"Pertanyaan: {question}\nJawaban:"
    
    print(f"\n🤔 Pertanyaan: {question}")
    print("⏳ Berpikir...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,           # Lebih pendek lagi
            min_new_tokens=5,            # Minimal 5 token
            temperature=0.3,             # Sangat deterministik
            top_p=0.75,                  # Lebih fokus
            top_k=30,                    # Sangat selektif
            do_sample=True,
            repetition_penalty=1.8,      # Tinggi untuk hindari repetisi
            no_repeat_ngram_size=4,      # Hindari 4-gram berulang
            length_penalty=0.5,          # Prefer jawaban pendek
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode hanya bagian jawaban (skip prompt)
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Minimal cleaning: stop di marker "Pertanyaan:" saja
    if "Pertanyaan:" in answer:
        answer = answer.split("Pertanyaan:")[0].strip()
    
    return answer


def demo_questions(model, tokenizer, device):
    """Demo dengan beberapa pertanyaan contoh"""
    
    print("\n" + "=" * 60)
    print("🎯 DEMO - Pertanyaan Contoh")
    print("=" * 60)
    
    questions = [
        "Apa ibu kota Indonesia?",
        "Siapa presiden pertama Indonesia?",
        "Apa itu komputer?",
        "Jelaskan tentang Pulau Jawa!",
    ]
    
    for question in questions:
        answer = ask_question(model, tokenizer, question, device)
        print(f"💬 Jawaban: {answer}")
        print("-" * 60)


def interactive_with_context(model, tokenizer, device):
    """Mode interaktif dengan konteks (simulasi RAG)"""
    
    # Contoh knowledge base tentang UUD 1945
    knowledge_base = {
        "uud45": """UUD 1945 adalah konstitusi negara Republik Indonesia. Disahkan pada tanggal 18 Agustus 1945. 
        UUD 1945 terdiri dari Pembukaan, Pasal-pasal, Aturan Peralihan, dan Aturan Tambahan.
        Pembukaan UUD 1945 berisi 4 alinea yang memuat dasar filosofis negara Indonesia, termasuk Pancasila.
        Pasal 1 ayat 1: Negara Indonesia ialah Negara Kesatuan yang berbentuk Republik.
        Pasal 27 ayat 1: Segala warga negara bersamaan kedudukannya di dalam hukum dan pemerintahan.
        UUD 1945 telah diamandemen sebanyak 4 kali antara tahun 1999-2002.""",
        
        "pancasila": """Pancasila adalah dasar negara Indonesia yang terdiri dari lima sila:
        1. Ketuhanan Yang Maha Esa
        2. Kemanusiaan yang Adil dan Beradab
        3. Persatuan Indonesia
        4. Kerakyatan yang Dipimpin oleh Hikmat Kebijaksanaan dalam Permusyawaratan/Perwakilan
        5. Keadilan Sosial bagi Seluruh Rakyat Indonesia.
        Pancasila dirumuskan oleh Ir. Soekarno pada tanggal 1 Juni 1945.""",
        
        "proklamasi": """Proklamasi Kemerdekaan Indonesia dibacakan pada tanggal 17 Agustus 1945 oleh Ir. Soekarno.
        Naskah proklamasi ditandatangani oleh Soekarno-Hatta atas nama bangsa Indonesia.
        Teks proklamasi: "Kami bangsa Indonesia dengan ini menjatakan kemerdekaan Indonesia.
        Hal-hal jang mengenai pemindahan kekoeasaan d.l.l., diselenggarakan dengan tjara saksama dan 
        dalam tempo jang sesingkat-singkatnja." """,
    }
    
    print("\n" + "=" * 60)
    print("🧠 MODE DENGAN KONTEKS (Simulasi RAG)")
    print("   Ketik pertanyaan Anda, atau 'quit' untuk keluar")
    print("   Ketik 'topik' untuk melihat topik yang tersedia")
    print("=" * 60)
    print("\n📚 Topik tersedia: uud45, pancasila, proklamasi")
    
    current_context = None
    current_topic = None
    
    while True:
        try:
            user_input = input("\n❓ Pertanyaan Anda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'keluar']:
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            
            if user_input.lower() == 'topik':
                print("\n📚 Topik tersedia:")
                for topic in knowledge_base.keys():
                    print(f"   - {topic}")
                continue
            
            if not user_input:
                continue
            
            # Skip pertanyaan yang terlalu pendek
            if len(user_input) < 3:
                print("💡 Pertanyaan terlalu pendek. Coba pertanyaan yang lebih spesifik.")
                continue
            
            # Check jika user menyebutkan topik tertentu
            question = user_input
            detected_topic = None
            
            for topic, context in knowledge_base.items():
                if topic in user_input.lower() or any(word in user_input.lower() for word in topic.split()):
                    detected_topic = topic
                    current_context = context
                    current_topic = topic
                    break
            
            if not question.endswith('?'):
                question += '?'
            
            # Tampilkan topik yang digunakan
            if current_context:
                print(f"📖 Menggunakan konteks: {current_topic}")
            else:
                print("💭 Menjawab tanpa konteks tambahan")
            
            answer = ask_question(model, tokenizer, question, device, context=current_context)
            print(f"💬 Jawaban: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih! Sampai jumpa!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def interactive_mode(model, tokenizer, device):
    """Mode interaktif - tanya jawab dengan model"""
    
    print("\n" + "=" * 60)
    print("🤖 MODE INTERAKTIF")
    print("   Ketik pertanyaan Anda, atau 'quit' untuk keluar")
    print("   Tips: Ajukan pertanyaan spesifik untuk hasil terbaik")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'keluar']:
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            
            if not question:
                continue
            
            # Skip pertanyaan yang terlalu pendek atau tidak jelas
            if len(question) < 3:
                print("💡 Pertanyaan terlalu pendek. Coba pertanyaan yang lebih spesifik.")
                continue
            
            # Pastikan ada tanda tanya di akhir pertanyaan
            if not question.endswith('?'):
                question += '?'
            
            answer = ask_question(model, tokenizer, question, device)
            print(f"💬 Jawaban: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih! Sampai jumpa!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    import sys
    
    print("=" * 60)
    print("🇮🇩 MASA AI - Q&A Model (150M Parameters)")
    print("   Model bahasa Indonesia untuk menjawab pertanyaan")
    print("=" * 60)
    
    # Load model
    try:
        model, tokenizer, device = load_masa_ai_model()
    except Exception as e:
        print("\n❌ Gagal memuat model. Pastikan dependencies terinstall:")
        print("   pip install -r requirements.txt")
        return
    
    # Show model info
    print(f"\n📊 Info Model:")
    print(f"   - Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"   - Max length: {tokenizer.model_max_length}")
    print(f"   - Device: {device}")
    
    # Check argument untuk mode
    mode = "demo"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--context":
            mode = "context"
        elif sys.argv[1] == "--interactive":
            mode = "interactive"
        elif sys.argv[1] == "--demo":
            mode = "demo"
    
    if mode == "context":
        # Mode dengan konteks (simulasi RAG)
        interactive_with_context(model, tokenizer, device)
    elif mode == "interactive":
        # Mode interaktif biasa
        interactive_mode(model, tokenizer, device)
    else:
        # Default: demo lalu interactive dengan konteks
        demo_questions(model, tokenizer, device)
        
        print("\n" + "=" * 60)
        print("Pilih mode:")
        print("  1. Mode Interaktif Biasa")
        print("  2. Mode dengan Konteks (RAG Simulation)")
        print("=" * 60)
        
        choice = input("\nPilihan (1/2): ").strip()
        
        if choice == "2":
            interactive_with_context(model, tokenizer, device)
        else:
            interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()

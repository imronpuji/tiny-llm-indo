"""
GPU Availability Check
======================
Check apakah GPU tersedia dan configured dengan benar
"""

import sys

print("=" * 70)
print("🔍 GPU AVAILABILITY CHECK")
print("=" * 70)

# 1. Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("\n  ⚠️  CUDA NOT AVAILABLE")
        print("\n  Possible reasons:")
        print("    1. PyTorch installed without CUDA support (CPU-only)")
        print("    2. No NVIDIA GPU detected")
        print("    3. CUDA drivers not installed")
        print("    4. Docker not running with --gpus flag")
        
        print("\n  Solutions:")
        print("    • Install PyTorch with CUDA:")
        print("      pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118")
        print("\n    • Or run Docker with GPU:")
        print("      docker run --gpus all -it your-container")

except ImportError:
    print("\n✗ PyTorch not installed!")
    sys.exit(1)

# 2. Check environment variables
print("\n" + "-" * 70)
print("🔧 ENVIRONMENT VARIABLES")
print("-" * 70)

import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")

if cuda_visible == '-1':
    print("  ⚠️  GPU explicitly disabled!")
    print("     Remove: unset CUDA_VISIBLE_DEVICES")
elif cuda_visible == 'not set':
    print("  ℹ️  Will use all available GPUs")

# 3. Test simple GPU operation
if torch.cuda.is_available():
    print("\n" + "-" * 70)
    print("🧪 GPU TEST")
    print("-" * 70)
    
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        
        print("  ✓ GPU computation successful!")
        print(f"  Device: {z.device}")
        
        # Memory info
        print(f"\n  GPU Memory:")
        print(f"    Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"    Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")

# 4. Recommendation
print("\n" + "=" * 70)
print("💡 RECOMMENDATION")
print("=" * 70)

if torch.cuda.is_available():
    print("  ✅ GPU is ready! Your model will use GPU automatically.")
    print(f"  Device: cuda (GPU {torch.cuda.current_device()})")
else:
    print("  ⚠️  No GPU available. Model will run on CPU (slower).")
    print("\n  For faster inference, install PyTorch with CUDA support:")
    print("    pip uninstall -y torch")
    print("    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118")
    
print("=" * 70)

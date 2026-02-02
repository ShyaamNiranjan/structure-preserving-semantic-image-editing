#!/usr/bin/env python3
"""
Fix script for CPU accelerate issues.
"""

import subprocess
import sys
import os

def update_accelerate():
    """Update accelerate to latest version."""
    try:
        print("🔄 Updating accelerate library...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "accelerate>=0.20.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Accelerate updated successfully")
            return True
        else:
            print(f"❌ Failed to update accelerate: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error updating accelerate: {e}")
        return False

def test_accelerate():
    """Test accelerate installation."""
    try:
        import accelerate
        print(f"✅ Accelerate version: {accelerate.__version__}")
        
        # Test CPU offload availability
        from accelerate.utils import is_accelerate_available
        print(f"✅ Accelerate available: {is_accelerate_available()}")
        
        return True
    except ImportError as e:
        print(f"❌ Accelerate not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing accelerate: {e}")
        return False

def main():
    """Main fix function."""
    print("🔧 Fixing CPU accelerate issues...")
    
    # Step 1: Update accelerate
    if not update_accelerate():
        print("⚠️ Could not update accelerate, continuing anyway...")
    
    # Step 2: Test accelerate
    if not test_accelerate():
        print("⚠️ Accelerate test failed, but system may still work")
    
    print("\n📝 Next steps:")
    print("1. Restart your backend server: cd backend && python main.py")
    print("2. Try the CPU test: python test_cpu_support.py")
    print("3. Test your image editing in the web interface")
    
    print("\n💡 If issues persist:")
    print("- The system will work without CPU offload (just uses more memory)")
    print("- Try installing CPU-only PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

if __name__ == "__main__":
    main()

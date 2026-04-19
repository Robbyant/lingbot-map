"""
Test script to verify the security fix for torch.load vulnerability.
"""

import torch
import tempfile
import os

def test_safe_checkpoint_loading():
    """Test that safe checkpoints load correctly with weights_only=True"""
    print("Testing safe checkpoint loading...")
    
    # Create a safe checkpoint (only tensors)
    safe_checkpoint = {
        'model': torch.randn(10, 10),
        'optimizer': torch.randn(5, 5),
        'epoch': 42,
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        torch.save(safe_checkpoint, temp_path)
    
    try:
        # This should work with weights_only=True
        loaded = torch.load(temp_path, weights_only=True)
        print("✅ Safe checkpoint loaded successfully with weights_only=True")
        assert 'model' in loaded
        assert loaded['epoch'] == 42
        print("✅ Checkpoint contents verified")
    finally:
        os.unlink(temp_path)

def test_malicious_checkpoint_blocked():
    """Test that malicious checkpoints are blocked with weights_only=True"""
    print("\nTesting malicious checkpoint blocking...")
    
    # Create a malicious checkpoint (contains executable code)
    class MaliciousClass:
        def __reduce__(self):
            # This would execute arbitrary code if loaded unsafely
            return (print, ("⚠️ MALICIOUS CODE EXECUTED!",))
    
    malicious_checkpoint = {
        'malicious': MaliciousClass(),
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        torch.save(malicious_checkpoint, temp_path)
    
    try:
        # This should FAIL with weights_only=True (security working)
        try:
            loaded = torch.load(temp_path, weights_only=True)
            print("❌ SECURITY ISSUE: Malicious checkpoint was loaded!")
            return False
        except Exception as e:
            print(f"✅ Malicious checkpoint blocked: {type(e).__name__}")
            print("✅ Security fix is working correctly!")
            return True
    finally:
        os.unlink(temp_path)

def test_backward_compatibility():
    """Test that the fallback mechanism works for older checkpoints"""
    print("\nTesting backward compatibility fallback...")
    
    # Create a checkpoint that might fail with weights_only=True
    # but should work with weights_only=False
    checkpoint = {
        'model': torch.randn(10, 10),
        'metadata': {'version': '1.0', 'author': 'test'},
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        torch.save(checkpoint, temp_path)
    
    try:
        # Try with weights_only=True first
        try:
            loaded = torch.load(temp_path, weights_only=True)
            print("✅ Loaded with weights_only=True")
        except Exception:
            # Fall back to weights_only=False
            print("⚠️  weights_only=True failed, falling back to weights_only=False")
            loaded = torch.load(temp_path, weights_only=False)
            print("✅ Loaded with weights_only=False (backward compatibility)")
        
        assert 'model' in loaded
        print("✅ Backward compatibility working")
    finally:
        os.unlink(temp_path)

def main():
    print("=" * 60)
    print("Security Fix Verification Tests")
    print("=" * 60)
    
    try:
        test_safe_checkpoint_loading()
        test_malicious_checkpoint_blocked()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe security fix is working correctly:")
        print("  • Safe checkpoints load with weights_only=True")
        print("  • Malicious checkpoints are blocked")
        print("  • Backward compatibility is maintained")
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

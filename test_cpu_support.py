#!/usr/bin/env python3
"""
Test script to verify CPU support for the diffusion engine.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cpu_detection():
    """Test CPU detection and configuration."""
    try:
        from config.settings import DEVICE, CUDA_AVAILABLE
        
        logger.info(f"🔍 Device Detection Results:")
        logger.info(f"   Device: {DEVICE}")
        logger.info(f"   CUDA Available: {CUDA_AVAILABLE}")
        
        if DEVICE == "cpu":
            logger.info("✅ CPU mode detected - system will run on CPU")
        else:
            logger.info("✅ GPU mode detected - system will run on GPU")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CPU detection test failed: {str(e)}")
        return False

def test_diffusion_engine_cpu():
    """Test diffusion engine initialization on CPU."""
    try:
        from ml_modules.inference.diffusion_engine import DiffusionEngine
        
        logger.info("🧪 Testing diffusion engine initialization...")
        
        # This will test if the model can be loaded on CPU
        engine = DiffusionEngine()
        
        # Check if pipeline was loaded
        if engine.pipeline is not None:
            logger.info(f"✅ Diffusion engine loaded successfully on {engine.device}")
            
            # Test model info
            model_info = engine.get_model_info()
            logger.info(f"   Model: {model_info['model_name']}")
            logger.info(f"   Pipeline: {model_info['pipeline_type']}")
            logger.info(f"   Device: {model_info['device']}")
            
            return True
        else:
            logger.error("❌ Diffusion engine pipeline not loaded")
            return False
        
    except Exception as e:
        logger.error(f"❌ Diffusion engine test failed: {str(e)}")
        return False

def test_cpu_optimization():
    """Test CPU-specific optimizations."""
    try:
        from config.settings import CPU_INFERENCE_STEPS, CPU_GUIDANCE_SCALE
        
        logger.info("🐌 Testing CPU optimization settings:")
        logger.info(f"   CPU Inference Steps: {CPU_INFERENCE_STEPS}")
        logger.info(f"   CPU Guidance Scale: {CPU_GUIDANCE_SCALE}")
        
        # Verify reasonable CPU settings
        if CPU_INFERENCE_STEPS <= 15:
            logger.info("✅ CPU inference steps are optimized for performance")
        else:
            logger.warning("⚠️ CPU inference steps might be too high for good performance")
        
        if CPU_GUIDANCE_SCALE <= 7.0:
            logger.info("✅ CPU guidance scale is optimized")
        else:
            logger.warning("⚠️ CPU guidance scale might be too high")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CPU optimization test failed: {str(e)}")
        return False

def main():
    """Run all CPU support tests."""
    logger.info("🖥️ Running CPU Support Tests...")
    
    tests = [
        ("CPU Detection", test_cpu_detection),
        ("CPU Optimization", test_cpu_optimization),
        ("Diffusion Engine CPU", test_diffusion_engine_cpu),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test {test_name} failed!")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All CPU support tests passed!")
        logger.info("💡 Your system should now work on CPU-only machines.")
        logger.info("⏱️ Note: CPU inference will be slower than GPU (expect 2-5 minutes per edit)")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

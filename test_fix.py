#!/usr/bin/env python3
"""
Test script to verify the structure extraction fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_structure_extractor():
    """Test the structure extractor with a sample image."""
    try:
        from ml_modules.perception.structure_extractor import StructureExtractor
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color='red')
        test_path = 'test_image.png'
        test_image.save(test_path)
        
        # Initialize structure extractor
        extractor = StructureExtractor()
        
        # Test structure extraction
        logger.info("Testing structure extraction...")
        structural_info = extractor.extract_structure(test_path)
        
        # Verify the results
        assert 'edges' in structural_info
        assert 'contours' in structural_info
        assert 'segments' in structural_info
        assert 'image_shape' in structural_info
        
        # Check that edges is a dictionary with expected keys
        assert isinstance(structural_info['edges'], dict)
        assert 'canny' in structural_info['edges']
        assert 'sobel' in structural_info['edges']
        assert 'laplacian' in structural_info['edges']
        
        # Check that contours is properly formatted
        assert isinstance(structural_info['contours'], dict)
        assert 'contours' in structural_info['contours']
        assert 'num_contours' in structural_info['contours']
        
        logger.info("✅ Structure extraction test passed!")
        
        # Clean up
        os.remove(test_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Structure extraction test failed: {str(e)}")
        return False

def test_diffusion_engine():
    """Test the diffusion engine initialization."""
    try:
        from ml_modules.inference.diffusion_engine import DiffusionEngine
        
        logger.info("Testing diffusion engine initialization...")
        
        # This will test if the model can be loaded (may fail if no GPU/internet)
        engine = DiffusionEngine()
        
        # Check if pipeline was loaded (may be None if model not available)
        if engine.pipeline is not None:
            logger.info("✅ Diffusion engine loaded successfully!")
        else:
            logger.warning("⚠️ Diffusion engine pipeline not loaded (expected if no GPU/model)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diffusion engine test failed: {str(e)}")
        return False

def test_imports():
    """Test if all modules can be imported."""
    try:
        logger.info("Testing imports...")
        
        from ml_modules.perception.structure_extractor import StructureExtractor
        from ml_modules.controller.structure_controller import StructureController
        from ml_modules.evaluation.metrics_calculator import MetricsCalculator
        from ml_modules.inference.diffusion_engine import DiffusionEngine
        
        logger.info("✅ All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running structure extraction fixes test...")
    
    tests = [
        ("Imports", test_imports),
        ("Structure Extractor", test_structure_extractor),
        ("Diffusion Engine", test_diffusion_engine),
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
        logger.info("🎉 All tests passed! The fixes should resolve the OpenCV error.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

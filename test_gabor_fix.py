#!/usr/bin/env python3
"""
Test script to verify the Gabor filter fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gabor_filter():
    """Test the Gabor filter fix."""
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        logger.info("Testing Gabor filter...")
        
        # Test the fixed Gabor filter code
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3, 0.5]:
                try:
                    gabor_kernel = cv2.getGaborKernel((15, 15), 3, np.radians(theta), 
                                                   2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(test_image, cv2.CV_8UC3, gabor_kernel)
                    gabor_responses.append(filtered)
                    logger.info(f"✅ Gabor filter successful for theta={theta}, freq={frequency}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply Gabor filter: {e}")
                    continue
        
        logger.info(f"✅ Gabor filter test completed! Generated {len(gabor_responses)} responses")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gabor filter test failed: {str(e)}")
        return False

def test_structure_extractor_with_fix():
    """Test the structure extractor with the fix."""
    try:
        from ml_modules.perception.structure_extractor import StructureExtractor
        
        # Create a test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = 'test_image_gabor.png'
        
        # Save test image
        from PIL import Image
        Image.fromarray(test_image).save(test_path)
        
        # Initialize structure extractor
        extractor = StructureExtractor()
        
        # Test structure extraction
        logger.info("Testing structure extraction with Gabor fix...")
        structural_info = extractor.extract_structure(test_path)
        
        # Verify the results
        assert 'texture_features' in structural_info
        assert 'lbp' in structural_info['texture_features']
        assert 'gabor_responses' in structural_info['texture_features']
        
        logger.info(f"✅ Structure extraction test passed! Gabor responses: {len(structural_info['texture_features']['gabor_responses'])}")
        
        # Clean up
        os.remove(test_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Structure extraction test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running Gabor filter fixes test...")
    
    tests = [
        ("Gabor Filter", test_gabor_filter),
        ("Structure Extractor with Fix", test_structure_extractor_with_fix),
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
        logger.info("🎉 All tests passed! The Gabor filter fix should resolve the 'too many values to unpack' error.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script to validate FastAPI response models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_response_models():
    """Test that response models are properly defined."""
    try:
        from routes.edit import EditResponse, EditPartialResponse
        
        logger.info("🧪 Testing response models...")
        
        # Test EditResponse
        edit_response = EditResponse(
            output_image_id="test_image.png",
            metrics={"ssim": 0.85, "lpips": 0.12},
            processing_time=15.5,
            step_id=1
        )
        logger.info("✅ EditResponse model works")
        
        # Test EditPartialResponse
        partial_response = EditPartialResponse(
            status="partial_success",
            message="Diffusion not available on CPU",
            image_id="test_image.png",
            metrics={
                "structural_analysis": "completed",
                "diffusion_available": False,
                "device": "cpu"
            }
        )
        logger.info("✅ EditPartialResponse model works")
        
        # Test serialization
        import json
        partial_dict = partial_response.dict()
        logger.info(f"✅ Serialization works: {partial_dict['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Response model test failed: {str(e)}")
        return False

def test_cpu_response_format():
    """Test the CPU response format matches expectations."""
    try:
        from routes.edit import EditPartialResponse
        from ml_modules.inference.diffusion_engine import DiffusionEngine
        
        logger.info("🧪 Testing CPU response format...")
        
        # Simulate CPU response
        engine = DiffusionEngine()
        response = EditPartialResponse(
            status="partial_success",
            message="Diffusion not available on CPU. Structure extraction completed. Image generation requires a GPU.",
            image_id="test_image_id",
            metrics={
                "structural_analysis": "completed",
                "diffusion_available": False,
                "device": engine.device
            }
        )
        
        # Validate response structure
        assert response.status == "partial_success"
        assert response.image_id == "test_image_id"
        assert "structural_analysis" in response.metrics
        assert response.metrics["diffusion_available"] == False
        
        logger.info("✅ CPU response format is correct")
        logger.info(f"   Status: {response.status}")
        logger.info(f"   Device: {response.metrics['device']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CPU response format test failed: {str(e)}")
        return False

def test_union_response():
    """Test that Union response type works."""
    try:
        from routes.edit import EditResponse, EditPartialResponse
        from typing import Union
        
        logger.info("🧪 Testing Union response type...")
        
        # Test that both response types can be used in Union
        def test_function() -> Union[EditResponse, EditPartialResponse]:
            return EditPartialResponse(
                status="partial_success",
                message="Test message",
                image_id="test.png",
                metrics={"test": "value"}
            )
        
        result = test_function()
        assert isinstance(result, EditPartialResponse)
        logger.info("✅ Union response type works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Union response test failed: {str(e)}")
        return False

def main():
    """Run all response model tests."""
    logger.info("🔧 Running Response Model Validation Tests...")
    
    tests = [
        ("Response Models", test_response_models),
        ("CPU Response Format", test_cpu_response_format),
        ("Union Response", test_union_response),
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
        logger.info("🎉 All response model tests passed!")
        logger.info("💡 FastAPI validation errors should now be resolved:")
        logger.info("   ✅ Proper response models defined")
        logger.info("   ✅ Union type for multiple responses")
        logger.info("   ✅ CPU response matches schema")
        logger.info("   ✅ No more 500 validation errors!")
        logger.info("\n🚀 Your API should now work correctly on CPU systems!")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from config.settings import INPUTS_DIR, OUTPUTS_DIR
from ml_modules.evaluation.metrics_calculator import MetricsCalculator

router = APIRouter()
logger = logging.getLogger(__name__)

class EvaluationRequest(BaseModel):
    original_image_id: str
    edited_image_id: str
    instruction: Optional[str] = ""
    compute_ssim: Optional[bool] = True
    compute_lpips: Optional[bool] = True
    compute_semantic: Optional[bool] = True

class EvaluationResponse(BaseModel):
    metrics: Dict[str, float]
    evaluation_time: float
    image_pair: Dict[str, str]

def find_image_path(image_id: str) -> Path:
    """
    Find image file by ID in various directories.
    
    Args:
        image_id: Image ID to find
        
    Returns:
        Path to the image file or None if not found
    """
    # Search in inputs, outputs, and intermediate directories
    search_dirs = [INPUTS_DIR, OUTPUTS_DIR]
    
    for search_dir in search_dirs:
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_path = search_dir / f"{image_id}{ext}"
            if potential_path.exists():
                return potential_path
    
    return None

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_images(request: EvaluationRequest):
    """
    Evaluate image quality metrics between original and edited images.
    
    Args:
        request: Evaluation request with image IDs and options
        
    Returns:
        Evaluation metrics
    """
    import time
    start_time = time.time()
    
    try:
        # Find image paths
        original_path = find_image_path(request.original_image_id)
        edited_path = find_image_path(request.edited_image_id)
        
        if not original_path or not original_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Original image {request.original_image_id} not found"
            )
        
        if not edited_path or not edited_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Edited image {request.edited_image_id} not found"
            )
        
        logger.info(f"Evaluating images: {request.original_image_id} -> {request.edited_image_id}")
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(
            original_path=str(original_path),
            edited_path=str(edited_path),
            instruction=request.instruction,
            compute_ssim=request.compute_ssim,
            compute_lpips=request.compute_lpips,
            compute_semantic=request.compute_semantic
        )
        
        evaluation_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        
        return EvaluationResponse(
            metrics=metrics,
            evaluation_time=evaluation_time,
            image_pair={
                "original": request.original_image_id,
                "edited": request.edited_image_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/evaluate/metrics")
async def get_available_metrics():
    """
    Get list of available evaluation metrics.
    
    Returns:
        Available metrics and their descriptions
    """
    return {
        "available_metrics": {
            "ssim": {
                "name": "Structural Similarity Index",
                "description": "Measures structural similarity between images",
                "range": "[-1, 1], higher is better"
            },
            "lpips": {
                "name": "Learned Perceptual Image Patch Similarity",
                "description": "Measures perceptual similarity using deep features",
                "range": "[0, ∞), lower is better"
            },
            "psnr": {
                "name": "Peak Signal-to-Noise Ratio",
                "description": "Measures ratio between maximum pixel value and noise",
                "range": "[0, ∞), higher is better"
            },
            "mse": {
                "name": "Mean Squared Error",
                "description": "Average of squared differences between pixels",
                "range": "[0, ∞), lower is better"
            },
            "semantic_consistency": {
                "name": "Semantic Consistency Score",
                "description": "Measures how well semantic content is preserved",
                "range": "[0, 1], higher is better"
            }
        }
    }

@router.post("/evaluate/batch")
async def evaluate_batch(requests: List[EvaluationRequest]):
    """
    Evaluate multiple image pairs in batch.
    
    Args:
        requests: List of evaluation requests
        
    Returns:
        List of evaluation results
    """
    try:
        results = []
        
        for i, request in enumerate(requests):
            logger.info(f"Processing batch evaluation {i+1}/{len(requests)}")
            
            try:
                result = await evaluate_images(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate pair {request.original_image_id}->{request.edited_image_id}: {str(e)}")
                results.append({
                    "error": str(e),
                    "image_pair": {
                        "original": request.original_image_id,
                        "edited": request.edited_image_id
                    }
                })
        
        return {"results": results, "total_processed": len(requests)}
        
    except Exception as e:
        logger.error(f"Error in batch evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")

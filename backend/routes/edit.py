from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Union, Dict, Any
import uuid
import logging
from pathlib import Path
from typing import Optional

from config.settings import OUTPUTS_DIR, INTERMEDIATE_DIR, INPUTS_DIR
from ml_modules.inference.diffusion_engine import DiffusionEngine
from ml_modules.perception.structure_extractor import StructureExtractor
from ml_modules.controller.structure_controller import StructureController
from ml_modules.evaluation.metrics_calculator import MetricsCalculator

router = APIRouter()
logger = logging.getLogger(__name__)

class EditRequest(BaseModel):
    image_id: str
    instruction_text: str
    step_id: Optional[int] = 0
    strength: Optional[float] = 0.8
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 20

class EditResponse(BaseModel):
    output_image_id: str
    metrics: Dict[str, float]
    processing_time: float
    step_id: int

class EditPartialResponse(BaseModel):
    status: str
    message: str
    image_id: str
    metrics: Dict[str, Any]

# Initialize ML components (lazy loading)
_diffusion_engine = None
_structure_extractor = None
_structure_controller = None
_metrics_calculator = None

def get_diffusion_engine():
    global _diffusion_engine
    if _diffusion_engine is None:
        _diffusion_engine = DiffusionEngine()
    return _diffusion_engine

def get_structure_extractor():
    global _structure_extractor
    if _structure_extractor is None:
        _structure_extractor = StructureExtractor()
    return _structure_extractor

def get_structure_controller():
    global _structure_controller
    if _structure_controller is None:
        _structure_controller = StructureController()
    return _structure_controller

def get_metrics_calculator():
    global _metrics_calculator
    if _metrics_calculator is None:
        _metrics_calculator = MetricsCalculator()
    return _metrics_calculator

@router.post("/edit", response_model=Union[EditResponse, EditPartialResponse])
async def apply_edit(request: EditRequest):
    """
    Apply structure-preserving edit to an image.
    
    Args:
        request: Edit request with image_id and instruction
        
    Returns:
        EditResponse with output image_id and metrics
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input image exists
        input_image_path = find_image_path(request.image_id)
        if not input_image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Input image {request.image_id} not found"
            )
        
        logger.info(f"Starting edit for image {request.image_id}: '{request.instruction_text}'")
        
        # Step 1: Extract structural information from input image
        structure_extractor = get_structure_extractor()
        structural_info = structure_extractor.extract_structure(str(input_image_path))
        
        # Step 2: Generate edited image using diffusion model
        diffusion_engine = get_diffusion_engine()
        
        # Check if diffusion is available (GPU required)
        if not diffusion_engine.loaded:
            logger.info("Diffusion not available on CPU - returning structure analysis only")
            
            # Return partial success with proper response model
            return EditPartialResponse(
                status="partial_success",
                message="Diffusion not available on CPU. Structure extraction completed. Image generation requires a GPU.",
                image_id=request.image_id,
                metrics={
                    "structural_analysis": "completed",
                    "diffusion_available": False,
                    "device": diffusion_engine.device
                }
            )
        
        output_image_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"
        
        generated_image = diffusion_engine.generate_edit(
            input_image_path=str(input_image_path),
            instruction=request.instruction_text,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            output_path=str(output_image_path)
        )
        
        # Step 3: Apply structure preservation constraints
        structure_controller = get_structure_controller()
        constrained_image = structure_controller.apply_constraints(
            generated_image=generated_image,
            structural_info=structural_info,
            output_path=str(output_image_path)
        )
        
        # Step 4: Calculate evaluation metrics
        metrics_calculator = get_metrics_calculator()
        metrics = metrics_calculator.calculate_metrics(
            original_path=str(input_image_path),
            edited_path=str(output_image_path),
            instruction=request.instruction_text
        )
        
        # Step 5: Save intermediate result for incremental editing
        intermediate_path = INTERMEDIATE_DIR / f"{request.image_id}_step_{request.step_id}.png"
        if constrained_image is not None:
            constrained_image.save(intermediate_path)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Edit completed in {processing_time:.2f}s for image {request.image_id}")
        
        return EditResponse(
            output_image_id=output_image_path.stem,
            metrics=metrics,
            processing_time=processing_time,
            step_id=request.step_id
        )
        
    except Exception as e:
        logger.error(f"Error applying edit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Edit failed: {str(e)}")

def find_image_path(image_id: str) -> Path:
    """
    Find image file by ID in various directories.
    
    Args:
        image_id: Image ID to find
        
    Returns:
        Path to the image file
    """
    # Search in inputs, outputs, and intermediate directories
    search_dirs = [INPUTS_DIR, OUTPUTS_DIR, INTERMEDIATE_DIR]
    
    for search_dir in search_dirs:
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_path = search_dir / f"{image_id}{ext}"
            if potential_path.exists():
                return potential_path
    
    return None

@router.get("/edit/status")
async def get_edit_status():
    """
    Get status of the editing system.
    
    Returns:
        System status information
    """
    return {
        "status": "ready",
        "models_loaded": {
            "diffusion_engine": _diffusion_engine is not None,
            "structure_extractor": _structure_extractor is not None,
            "structure_controller": _structure_controller is not None,
            "metrics_calculator": _metrics_calculator is not None
        }
    }

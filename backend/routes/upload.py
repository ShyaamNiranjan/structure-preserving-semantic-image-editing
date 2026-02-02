from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
import io
from pathlib import Path
from PIL import Image
import logging

from config.settings import INPUTS_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for processing.
    
    Args:
        file: Image file to upload
        
    Returns:
        JSON response with image_id
    """
    try:
        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        filename = f"{image_id}{file_extension}"
        file_path = INPUTS_DIR / filename
        
        # Validate image content
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image
            image = Image.open(io.BytesIO(content))  # Reopen after verify
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Save the image
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Get image metadata
        width, height = image.size
        mode = image.mode
        
        logger.info(f"Image uploaded: {image_id}, size: {width}x{height}, mode: {mode}")
        
        return {
            "image_id": image_id,
            "filename": filename,
            "width": width,
            "height": height,
            "mode": mode,
            "file_size": file_size,
            "message": "Image uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/upload/limits")
async def get_upload_limits():
    """
    Get upload limits and allowed file types.
    
    Returns:
        JSON with upload configuration
    """
    return {
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
    }

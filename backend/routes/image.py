from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

from config.settings import INPUTS_DIR, OUTPUTS_DIR, INTERMEDIATE_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    Retrieve an image by ID.
    
    Args:
        image_id: ID of the image to retrieve
        
    Returns:
        Image file as response
    """
    try:
        # Search for image in all directories
        image_path = find_image_path(image_id)
        
        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image {image_id} not found"
            )
        
        logger.info(f"Serving image: {image_id}")
        
        return FileResponse(
            path=str(image_path),
            media_type=f"image/{image_path.suffix[1:]}",
            filename=image_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")

@router.get("/image/{image_id}/info")
async def get_image_info(image_id: str):
    """
    Get metadata about an image.
    
    Args:
        image_id: ID of the image
        
    Returns:
        Image metadata
    """
    try:
        image_path = find_image_path(image_id)
        
        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image {image_id} not found"
            )
        
        # Get image metadata
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
        
        file_size = image_path.stat().st_size
        
        return {
            "image_id": image_id,
            "filename": image_path.name,
            "width": width,
            "height": height,
            "mode": mode,
            "format": format_name,
            "file_size": file_size,
            "directory": image_path.parent.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image info {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get image info: {str(e)}")

def find_image_path(image_id: str) -> Path:
    """
    Find image file by ID in various directories.
    
    Args:
        image_id: Image ID to find
        
    Returns:
        Path to the image file or None if not found
    """
    # Search in inputs, outputs, and intermediate directories
    search_dirs = [INPUTS_DIR, OUTPUTS_DIR, INTERMEDIATE_DIR]
    
    for search_dir in search_dirs:
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_path = search_dir / f"{image_id}{ext}"
            if potential_path.exists():
                return potential_path
    
    return None

@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """
    Delete an image by ID.
    
    Args:
        image_id: ID of the image to delete
        
    Returns:
        Deletion status
    """
    try:
        image_path = find_image_path(image_id)
        
        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image {image_id} not found"
            )
        
        image_path.unlink()
        logger.info(f"Deleted image: {image_id}")
        
        return {"message": f"Image {image_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

import cv2
import numpy as np
from PIL import Image
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy import ndimage
from skimage import restoration, morphology

logger = logging.getLogger(__name__)

class StructureController:
    """
    Applies structure preservation constraints during image generation.
    
    This class handles:
    - Region-based constraints
    - Edge preservation
    - Incremental editing logic
    - Structural consistency enforcement
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.constraint_strength = 0.7  # Default constraint strength
        
    def apply_constraints(
        self,
        generated_image: Image.Image,
        structural_info: Dict[str, Any],
        original_image: Optional[Image.Image] = None,
        output_path: Optional[str] = None,
        constraint_strength: float = 0.7
    ) -> Image.Image:
        """
        Apply structure preservation constraints to generated image.
        
        Args:
            generated_image: Image from diffusion model
            structural_info: Structural information from perception module
            original_image: Original input image for reference
            output_path: Path to save constrained image
            constraint_strength: Strength of constraints (0-1)
            
        Returns:
            Constrained PIL Image
        """
        try:
            logger.info(f"Applying structure constraints with strength {constraint_strength}")
            
            # Convert to numpy array
            generated_np = np.array(generated_image)
            
            # Generate preservation mask
            preservation_mask = self._generate_preservation_mask(
                structural_info, constraint_strength
            )
            
            # Apply edge preservation
            edge_preserved = self._preserve_edges(
                generated_np, structural_info, preservation_mask
            )
            
            # Apply region constraints
            region_constrained = self._apply_region_constraints(
                edge_preserved, structural_info, preservation_mask
            )
            
            # Apply incremental editing logic if original image provided
            if original_image:
                final_image = self._apply_incremental_logic(
                    region_constrained, original_image, structural_info
                )
            else:
                final_image = region_constrained
            
            # Convert back to PIL Image
            result_image = Image.fromarray(final_image.astype(np.uint8))
            
            # Save if path provided
            if output_path:
                result_image.save(output_path)
                logger.info(f"Constrained image saved to: {output_path}")
            
            logger.info("Structure constraints applied successfully")
            return result_image
            
        except Exception as e:
            logger.error(f"Failed to apply constraints: {str(e)}")
            raise
    
    def _generate_preservation_mask(
        self, 
        structural_info: Dict[str, Any], 
        constraint_strength: float
    ) -> np.ndarray:
        """Generate mask for structure preservation."""
        try:
            image_shape = structural_info["image_shape"][:2]
            mask = np.zeros(image_shape, dtype=np.float32)
            
            # Add edge information
            edges = structural_info["edges"]["canny"]
            mask[edges > 0] = 1.0
            
            # Add contour regions
            for contour_info in structural_info["contours"]["contours"]:
                if contour_info["area"] > 100:
                    cv2.drawContours(mask, [contour_info["contour"]], -1, 1.0, 2)
            
            # Add semantic regions if available
            if structural_info.get("semantic_info"):
                for segment in structural_info["semantic_info"]["segments"]:
                    if segment.get("mask") is not None:
                        semantic_mask = np.array(segment["mask"])
                        mask[semantic_mask > 0] = np.maximum(
                            mask[semantic_mask > 0], 0.8
                        )
            
            # Apply constraint strength
            mask = mask * constraint_strength
            
            # Smooth the mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to generate preservation mask: {str(e)}")
            raise
    
    def _preserve_edges(
        self, 
        image: np.ndarray, 
        structural_info: Dict[str, Any], 
        mask: np.ndarray
    ) -> np.ndarray:
        """Preserve important edges in the image."""
        try:
            # Extract edges from generated image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            current_edges = cv2.Canny(gray, 50, 150)
            
            # Get original edge information
            original_edges = structural_info["edges"]["canny"]
            
            # Blend edges based on mask
            preserved_edges = np.where(
                mask > 0.5,
                original_edges,
                current_edges
            )
            
            # Apply edge-preserving filter
            preserved_image = cv2.edgePreservingFilter(
                image, flags=1, sigma_s=50, sigma_r=0.4
            )
            
            # Enhance edges where needed
            edge_enhanced = self._enhance_edges(preserved_image, preserved_edges, mask)
            
            return edge_enhanced
            
        except Exception as e:
            logger.error(f"Failed to preserve edges: {str(e)}")
            return image
    
    def _enhance_edges(
        self, 
        image: np.ndarray, 
        target_edges: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """Enhance edges in specific regions."""
        try:
            # Create edge enhancement kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # Apply enhancement only where mask indicates
            enhanced = image.copy()
            for i in range(3):  # RGB channels
                channel = image[:, :, i]
                enhanced_channel = cv2.filter2D(channel, -1, kernel)
                enhanced[:, :, i] = np.where(
                    mask > 0.3,
                    np.clip(enhanced_channel, 0, 255),
                    channel
                )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to enhance edges: {str(e)}")
            return image
    
    def _apply_region_constraints(
        self, 
        image: np.ndarray, 
        structural_info: Dict[str, Any], 
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply region-based constraints."""
        try:
            constrained_image = image.copy()
            
            # Apply constraints based on segmentation
            segments = structural_info["segments"]["slic"]
            
            for segment_id in np.unique(segments):
                if segment_id == 0:
                    continue
                
                # Create mask for this segment
                segment_mask = (segments == segment_id)
                
                # Calculate average constraint strength for this segment
                avg_constraint = np.mean(mask[segment_mask])
                
                if avg_constraint > 0.5:
                    # Apply smoothing to preserve structure
                    segment_region = image * segment_mask[:, :, np.newaxis]
                    smoothed_segment = cv2.bilateralFilter(
                        segment_region.astype(np.uint8), 9, 75, 75
                    )
                    
                    # Blend based on constraint strength
                    constrained_image[segment_mask] = (
                        (1 - avg_constraint) * constrained_image[segment_mask] +
                        avg_constraint * smoothed_segment[segment_mask]
                    )
            
            return constrained_image
            
        except Exception as e:
            logger.error(f"Failed to apply region constraints: {str(e)}")
            return image
    
    def _apply_incremental_logic(
        self, 
        generated_image: np.ndarray, 
        original_image: Image.Image, 
        structural_info: Dict[str, Any]
    ) -> np.ndarray:
        """Apply incremental editing logic to maintain consistency."""
        try:
            original_np = np.array(original_image)
            
            # Ensure same size
            if generated_image.shape != original_np.shape:
                original_np = cv2.resize(original_np, 
                                       (generated_image.shape[1], generated_image.shape[0]))
            
            # Calculate structural similarity
            gray_gen = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
            gray_orig = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
            
            # Create adaptive blending mask based on structural changes
            structural_diff = np.abs(gray_gen.astype(float) - gray_orig.astype(float))
            change_mask = (structural_diff > np.mean(structural_diff) + np.std(structural_diff))
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            change_mask = cv2.morphologyEx(change_mask.astype(np.uint8), 
                                         cv2.MORPH_CLOSE, kernel)
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
            
            # Blend images based on change mask
            alpha = 0.7  # Weight for generated image
            final_image = np.where(
                change_mask[:, :, np.newaxis],
                generated_image,
                alpha * generated_image + (1 - alpha) * original_np
            )
            
            return final_image.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Failed to apply incremental logic: {str(e)}")
            return generated_image
    
    def validate_structure_preservation(
        self, 
        original_image: Image.Image, 
        edited_image: Image.Image,
        structural_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate how well structure was preserved.
        
        Args:
            original_image: Original input image
            edited_image: Final edited image
            structural_info: Original structural information
            
        Returns:
            Validation metrics
        """
        try:
            original_np = np.array(original_image)
            edited_np = np.array(edited_image)
            
            # Calculate edge preservation score
            orig_edges = cv2.Canny(cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY), 50, 150)
            edited_edges = cv2.Canny(cv2.cvtColor(edited_np, cv2.COLOR_RGB2GRAY), 50, 150)
            
            edge_preservation = np.sum(np.minimum(orig_edges, edited_edges)) / np.sum(orig_edges)
            
            # Calculate contour preservation
            orig_contours = structural_info["contours"]["contours"]
            preserved_contours = 0
            
            for contour_info in orig_contours:
                if contour_info["area"] > 100:
                    # Check if contour is still present
                    contour_mask = np.zeros(original_np.shape[:2], dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour_info["contour"]], -1, 1, 2)
                    
                    if np.sum(edited_edges[contour_mask > 0]) > 0:
                        preserved_contours += 1
            
            contour_preservation = preserved_contours / len([c for c in orig_contours if c["area"] > 100]) if orig_contours else 1.0
            
            # Calculate overall structure preservation score
            overall_score = (edge_preservation + contour_preservation) / 2
            
            return {
                "edge_preservation": edge_preservation,
                "contour_preservation": contour_preservation,
                "overall_preservation": overall_score
            }
            
        except Exception as e:
            logger.error(f"Failed to validate structure preservation: {str(e)}")
            return {
                "edge_preservation": 0.0,
                "contour_preservation": 0.0,
                "overall_preservation": 0.0
            }
    
    def adjust_constraint_strength(
        self, 
        current_score: float, 
        target_score: float = 0.8
    ) -> float:
        """
        Dynamically adjust constraint strength based on preservation score.
        
        Args:
            current_score: Current structure preservation score
            target_score: Target preservation score
            
        Returns:
            Adjusted constraint strength
        """
        try:
            if current_score >= target_score:
                return self.constraint_strength
            
            # Increase constraint strength if preservation is low
            adjustment = (target_score - current_score) * 0.5
            new_strength = min(1.0, self.constraint_strength + adjustment)
            
            logger.info(f"Adjusting constraint strength: {self.constraint_strength:.3f} -> {new_strength:.3f}")
            self.constraint_strength = new_strength
            
            return new_strength
            
        except Exception as e:
            logger.error(f"Failed to adjust constraint strength: {str(e)}")
            return self.constraint_strength

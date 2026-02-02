import cv2
import numpy as np
from PIL import Image
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from skimage import segmentation, filters, morphology
from transformers import pipeline as transformers_pipeline

logger = logging.getLogger(__name__)

class StructureExtractor:
    """
    Extracts structural and semantic information from images for guidance.
    
    This class handles:
    - Edge detection and boundary extraction
    - Semantic segmentation
    - Structural mask generation
    - Feature extraction for constraints
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmentation_model = None
        self._load_models()
    
    def _load_models(self):
        """Load perception models."""
        try:
            # Load semantic segmentation model
            # Using a lightweight model for research purposes
            logger.info("Loading perception models...")
            
            # Initialize segmentation pipeline (using a pre-trained model)
            try:
                self.segmentation_model = transformers_pipeline(
                    "image-segmentation",
                    model="facebook/detr-resnet-50-panoptic",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Semantic segmentation model loaded")
            except Exception as e:
                logger.warning(f"Failed to load segmentation model: {e}")
                logger.info("Falling back to traditional computer vision methods")
                self.segmentation_model = None
            
        except Exception as e:
            logger.error(f"Failed to load perception models: {str(e)}")
    
    def extract_structure(self, image_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive structural information from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing structural information
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract various structural features
            edges = self._extract_edges(gray)
            contours = self._extract_contours(edges["canny"])  # Pass canny edges specifically
            segments = self._extract_segments(image_rgb)
            semantic_info = self._extract_semantic_info(image_path) if self.segmentation_model else None
            
            # Combine all structural information
            structural_info = {
                "edges": edges,
                "contours": contours,
                "segments": segments,
                "semantic_info": semantic_info,
                "image_shape": image.shape,
                "gradients": self._extract_gradients(gray),
                "texture_features": self._extract_texture_features(gray)
            }
            
            logger.info(f"Structure extraction completed for {image_path}")
            return structural_info
            
        except Exception as e:
            logger.error(f"Failed to extract structure from {image_path}: {str(e)}")
            raise
    
    def _extract_edges(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract edge information using multiple methods."""
        try:
            # Canny edge detection
            canny_edges = cv2.Canny(gray_image, 50, 150)
            
            # Sobel edges
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Laplacian edges
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            
            return {
                "canny": canny_edges,
                "sobel": sobel_magnitude,
                "laplacian": np.abs(laplacian)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract edges: {str(e)}")
            raise
    
    def _extract_contours(self, edges: np.ndarray) -> Dict[str, Any]:
        """Extract contour information from edges."""
        try:
            # Ensure edges is a valid NumPy array
            if not isinstance(edges, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(edges)}")
            
            if edges.dtype != np.uint8:
                edges = edges.astype(np.uint8)
            
            # Find contours
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter significant contours
            significant_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    significant_contours.append({
                        "contour": contour,
                        "area": area,
                        "perimeter": cv2.arcLength(contour, True),
                        "bounding_box": cv2.boundingRect(contour)
                    })
            
            return {
                "contours": significant_contours,
                "num_contours": len(significant_contours),
                "hierarchy": hierarchy
            }
            
        except Exception as e:
            logger.error(f"Failed to extract contours: {str(e)}")
            # Return empty contours on error to prevent system crash
            return {
                "contours": [],
                "num_contours": 0,
                "hierarchy": None
            }
    
    def _extract_segments(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract image segments using various methods."""
        try:
            # Felzenszwalb segmentation
            segments_fz = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
            
            # SLIC superpixels
            segments_slic = segmentation.slic(image, n_segments=100, compactness=10, sigma=1)
            
            # Quickshift segmentation
            segments_quick = segmentation.quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
            
            return {
                "felzenszwalb": segments_fz,
                "slic": segments_slic,
                "quickshift": segments_quick,
                "num_segments_fz": len(np.unique(segments_fz)),
                "num_segments_slic": len(np.unique(segments_slic)),
                "num_segments_quick": len(np.unique(segments_quick))
            }
            
        except Exception as e:
            logger.error(f"Failed to extract segments: {str(e)}")
            raise
    
    def _extract_semantic_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract semantic information using segmentation model."""
        try:
            if not self.segmentation_model:
                return None
            
            # Use the segmentation model
            results = self.segmentation_model(image_path)
            
            # Process results
            semantic_info = {
                "segments": [],
                "num_objects": len(results) if isinstance(results, list) else 1
            }
            
            if isinstance(results, list):
                for result in results:
                    semantic_info["segments"].append({
                        "label": result.get("label", "unknown"),
                        "score": result.get("score", 0.0),
                        "mask": result.get("mask", None)
                    })
            
            return semantic_info
            
        except Exception as e:
            logger.warning(f"Failed to extract semantic info: {str(e)}")
            return None
    
    def _extract_gradients(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract gradient information."""
        try:
            # Gradient magnitude and direction
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_direction = np.arctan2(grad_y, grad_x)
            
            return {
                "magnitude": grad_magnitude,
                "direction": grad_direction,
                "x": grad_x,
                "y": grad_y
            }
            
        except Exception as e:
            logger.error(f"Failed to extract gradients: {str(e)}")
            raise
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract texture features."""
        try:
            # Local Binary Pattern
            from skimage.feature import local_binary_pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Gabor filters
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3, 0.5]:
                    try:
                        gabor_kernel = cv2.getGaborKernel((15, 15), 3, np.radians(theta), 
                                                       2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                        filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, gabor_kernel)
                        gabor_responses.append(filtered)
                    except Exception as e:
                        logger.warning(f"Failed to apply Gabor filter: {e}")
                        continue
            
            return {
                "lbp": lbp,
                "gabor_responses": gabor_responses
            }
            
        except Exception as e:
            logger.error(f"Failed to extract texture features: {str(e)}")
            # Return default texture features on error to prevent system crash
            return {
                "lbp": np.zeros_like(gray_image),
                "gabor_responses": []
            }
    
    def generate_preservation_mask(
        self, 
        structural_info: Dict[str, Any], 
        preserve_regions: Optional[list] = None
    ) -> np.ndarray:
        """
        Generate a mask for structure preservation.
        
        Args:
            structural_info: Structural information from extract_structure
            preserve_regions: List of region types to preserve
            
        Returns:
            Binary mask for preservation
        """
        try:
            image_shape = structural_info["image_shape"][:2]
            mask = np.zeros(image_shape, dtype=np.uint8)
            
            # Combine edges for preservation
            edges = structural_info["edges"]["canny"]
            mask[edges > 0] = 1
            
            # Add contour regions
            for contour_info in structural_info["contours"]["contours"]:
                if contour_info["area"] > 500:  # Preserve larger structures
                    cv2.drawContours(mask, [contour_info["contour"]], -1, 1, 2)
            
            # Dilate mask to include surrounding areas
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to generate preservation mask: {str(e)}")
            raise
    
    def get_structure_summary(self, structural_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of structural information.
        
        Args:
            structural_info: Structural information dictionary
            
        Returns:
            Summary statistics
        """
        try:
            summary = {
                "image_shape": structural_info["image_shape"],
                "num_contours": structural_info["contours"]["num_contours"],
                "num_segments_fz": structural_info["segments"]["num_segments_fz"],
                "edge_density": np.mean(structural_info["edges"]["canny"]),
                "gradient_magnitude_mean": np.mean(structural_info["gradients"]["magnitude"]),
                "has_semantic_info": structural_info["semantic_info"] is not None
            }
            
            if structural_info["semantic_info"]:
                summary["num_semantic_objects"] = structural_info["semantic_info"]["num_objects"]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate structure summary: {str(e)}")
            raise

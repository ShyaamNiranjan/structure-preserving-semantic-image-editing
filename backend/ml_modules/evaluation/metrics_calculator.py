import cv2
import numpy as np
from PIL import Image
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import cosine
import lpips

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculates various quality metrics for image evaluation.
    
    This class handles:
    - Structural similarity metrics (SSIM)
    - Perceptual similarity metrics (LPIPS)
    - Traditional metrics (PSNR, MSE)
    - Semantic consistency evaluation
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips_model = None
        self._load_models()
    
    def _load_models(self):
        """Load evaluation models."""
        try:
            # Initialize LPIPS model
            logger.info("Loading LPIPS model...")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            logger.info("LPIPS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LPIPS model: {str(e)}")
            self.lpips_model = None
    
    def calculate_metrics(
        self,
        original_path: str,
        edited_path: str,
        instruction: str = "",
        compute_ssim: bool = True,
        compute_lpips: bool = True,
        compute_semantic: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            instruction: Text instruction used for editing
            compute_ssim: Whether to compute SSIM
            compute_lpips: Whether to compute LPIPS
            compute_semantic: Whether to compute semantic consistency
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Load images
            original_image = self._load_image(original_path)
            edited_image = self._load_image(edited_path)
            
            # Ensure same size
            if original_image.shape != edited_image.shape:
                edited_image = cv2.resize(edited_image, 
                                        (original_image.shape[1], original_image.shape[0]))
            
            metrics = {}
            
            # Calculate SSIM
            if compute_ssim:
                metrics["ssim"] = self._calculate_ssim(original_image, edited_image)
            
            # Calculate PSNR
            metrics["psnr"] = self._calculate_psnr(original_image, edited_image)
            
            # Calculate MSE
            metrics["mse"] = self._calculate_mse(original_image, edited_image)
            
            # Calculate LPIPS
            if compute_lpips and self.lpips_model:
                metrics["lpips"] = self._calculate_lpips(original_image, edited_image)
            
            # Calculate semantic consistency
            if compute_semantic:
                metrics["semantic_consistency"] = self._calculate_semantic_consistency(
                    original_image, edited_image, instruction
                )
            
            # Calculate edge preservation
            metrics["edge_preservation"] = self._calculate_edge_preservation(
                original_image, edited_image
            )
            
            # Calculate color consistency
            metrics["color_consistency"] = self._calculate_color_consistency(
                original_image, edited_image
            )
            
            logger.info(f"Metrics calculated for {original_path} -> {edited_path}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            raise
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        try:
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Calculate SSIM
            ssim_value, _ = ssim(gray1, gray2, full=True)
            return float(ssim_value)
            
        except Exception as e:
            logger.error(f"Failed to calculate SSIM: {str(e)}")
            return 0.0
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        try:
            psnr_value = psnr(img1, img2)
            return float(psnr_value)
            
        except Exception as e:
            logger.error(f"Failed to calculate PSNR: {str(e)}")
            return 0.0
    
    def _calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        try:
            mse_value = np.mean((img1 - img2) ** 2)
            return float(mse_value)
            
        except Exception as e:
            logger.error(f"Failed to calculate MSE: {str(e)}")
            return float('inf')
    
    def _calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Learned Perceptual Image Patch Similarity."""
        try:
            if not self.lpips_model:
                return 0.0
            
            # Convert to tensor and normalize to [-1, 1]
            img1_tensor = self._image_to_tensor(img1)
            img2_tensor = self._image_to_tensor(img2)
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_value = self.lpips_model(img1_tensor, img2_tensor)
            
            return float(lpips_value.item())
            
        except Exception as e:
            logger.error(f"Failed to calculate LPIPS: {str(e)}")
            return 0.0
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor for LPIPS."""
        # Normalize to [0, 1] and then to [-1, 1]
        image_normalized = image.astype(np.float32) / 255.0
        image_normalized = image_normalized * 2.0 - 1.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _calculate_semantic_consistency(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray, 
        instruction: str
    ) -> float:
        """Calculate semantic consistency between images."""
        try:
            # Extract semantic features using simple methods
            # In a production system, you might use a pre-trained semantic segmentation model
            
            # Calculate color histogram similarity
            hist1 = self._calculate_color_histogram(img1)
            hist2 = self._calculate_color_histogram(img2)
            
            hist_similarity = 1.0 - cosine(hist1, hist2)
            
            # Calculate texture similarity
            texture_similarity = self._calculate_texture_similarity(img1, img2)
            
            # Combine features
            semantic_score = (hist_similarity + texture_similarity) / 2.0
            
            return float(semantic_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate semantic consistency: {str(e)}")
            return 0.0
    
    def _calculate_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Calculate color histogram for semantic analysis."""
        try:
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256])
            
            # Normalize and concatenate
            hist_r = hist_r.flatten() / hist_r.sum()
            hist_g = hist_g.flatten() / hist_g.sum()
            hist_b = hist_b.flatten() / hist_b.sum()
            
            return np.concatenate([hist_r, hist_g, hist_b])
            
        except Exception as e:
            logger.error(f"Failed to calculate color histogram: {str(e)}")
            return np.zeros(192)  # 64 * 3 channels
    
    def _calculate_texture_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate texture similarity using LBP features."""
        try:
            from skimage.feature import local_binary_pattern
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Calculate LBP
            radius = 3
            n_points = 8 * radius
            lbp1 = local_binary_pattern(gray1, n_points, radius, method='uniform')
            lbp2 = local_binary_pattern(gray2, n_points, radius, method='uniform')
            
            # Calculate histograms
            hist1, _ = np.histogram(lbp1.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist2, _ = np.histogram(lbp2.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            
            # Normalize
            hist1 = hist1.astype(float) / hist1.sum()
            hist2 = hist2.astype(float) / hist2.sum()
            
            # Calculate similarity
            similarity = 1.0 - cosine(hist1, hist2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate texture similarity: {str(e)}")
            return 0.0
    
    def _calculate_edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate how well edges are preserved."""
        try:
            # Extract edges
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # Calculate edge preservation
            intersection = np.sum(np.minimum(edges1, edges2))
            union = np.sum(np.maximum(edges1, edges2))
            
            if union == 0:
                return 1.0
            
            edge_preservation = intersection / union
            return float(edge_preservation)
            
        except Exception as e:
            logger.error(f"Failed to calculate edge preservation: {str(e)}")
            return 0.0
    
    def _calculate_color_consistency(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate color consistency between images."""
        try:
            # Calculate mean and std for each channel
            mean1 = np.mean(img1, axis=(0, 1))
            mean2 = np.mean(img2, axis=(0, 1))
            std1 = np.std(img1, axis=(0, 1))
            std2 = np.std(img2, axis=(0, 1))
            
            # Calculate consistency scores
            mean_consistency = 1.0 - np.mean(np.abs(mean1 - mean2)) / 255.0
            std_consistency = 1.0 - np.mean(np.abs(std1 - std2)) / 255.0
            
            overall_consistency = (mean_consistency + std_consistency) / 2.0
            return float(overall_consistency)
            
        except Exception as e:
            logger.error(f"Failed to calculate color consistency: {str(e)}")
            return 0.0
    
    def get_metric_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get a summary interpretation of metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Summary with interpretations
        """
        try:
            summary = {
                "metrics": metrics,
                "interpretations": {},
                "overall_quality": "unknown"
            }
            
            # Interpret SSIM
            if "ssim" in metrics:
                ssim_val = metrics["ssim"]
                if ssim_val > 0.9:
                    summary["interpretations"]["ssim"] = "excellent"
                elif ssim_val > 0.8:
                    summary["interpretations"]["ssim"] = "good"
                elif ssim_val > 0.6:
                    summary["interpretations"]["ssim"] = "fair"
                else:
                    summary["interpretations"]["ssim"] = "poor"
            
            # Interpret LPIPS
            if "lpips" in metrics:
                lpips_val = metrics["lpips"]
                if lpips_val < 0.1:
                    summary["interpretations"]["lpips"] = "excellent"
                elif lpips_val < 0.2:
                    summary["interpretations"]["lpips"] = "good"
                elif lpips_val < 0.4:
                    summary["interpretations"]["lpips"] = "fair"
                else:
                    summary["interpretations"]["lpips"] = "poor"
            
            # Calculate overall quality
            scores = []
            if "ssim" in metrics:
                scores.append(metrics["ssim"])  # Higher is better
            if "lpips" in metrics:
                scores.append(1.0 - min(metrics["lpips"], 1.0))  # Convert to higher is better
            if "edge_preservation" in metrics:
                scores.append(metrics["edge_preservation"])
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > 0.8:
                    summary["overall_quality"] = "excellent"
                elif avg_score > 0.6:
                    summary["overall_quality"] = "good"
                elif avg_score > 0.4:
                    summary["overall_quality"] = "fair"
                else:
                    summary["overall_quality"] = "poor"
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate metric summary: {str(e)}")
            return {"metrics": metrics, "interpretations": {}, "overall_quality": "unknown"}
    
    def benchmark_metrics(self, test_images_path: str) -> Dict[str, Any]:
        """
        Benchmark metrics calculation on test images.
        
        Args:
            test_images_path: Path to test images directory
            
        Returns:
            Benchmark results
        """
        try:
            test_dir = Path(test_images_path)
            if not test_dir.exists():
                raise ValueError(f"Test directory not found: {test_images_path}")
            
            # Find test image pairs (original and edited)
            image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
            
            benchmark_results = {
                "num_images": len(image_files),
                "metrics": [],
                "average_metrics": {},
                "processing_times": []
            }
            
            import time
            
            for i, img_path in enumerate(image_files):
                if i % 2 == 1:  # Skip every other image for now (assuming pairs)
                    continue
                
                start_time = time.time()
                
                # Use same image for testing (in real scenario, you'd have pairs)
                metrics = self.calculate_metrics(
                    str(img_path), str(img_path), "test"
                )
                
                processing_time = time.time() - start_time
                
                benchmark_results["metrics"].append(metrics)
                benchmark_results["processing_times"].append(processing_time)
            
            # Calculate averages
            if benchmark_results["metrics"]:
                all_metric_keys = benchmark_results["metrics"][0].keys()
                for key in all_metric_keys:
                    values = [m[key] for m in benchmark_results["metrics"]]
                    benchmark_results["average_metrics"][key] = np.mean(values)
                
                benchmark_results["avg_processing_time"] = np.mean(benchmark_results["processing_times"])
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark metrics: {str(e)}")
            raise

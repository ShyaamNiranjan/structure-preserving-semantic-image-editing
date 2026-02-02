import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any

from config.settings import DEVICE, MODEL_NAME, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, CPU_INFERENCE_STEPS, CPU_GUIDANCE_SCALE

logger = logging.getLogger(__name__)

class DiffusionEngine:
    """
    Core diffusion model engine for image-to-image generation.
    
    This class handles:
    - Loading and managing diffusion models
    - Text encoding and conditioning
    - Image-to-image generation with structural guidance
    """
    
    def __init__(self):
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self.pipeline = None
        self.tokenizer = None
        self.text_encoder = None
        self.loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the diffusion model and related components."""
        try:
            logger.info(f"Checking diffusion model availability on {self.device}")
            
            # Check if CUDA is available - diffusion requires CUDA
            if self.device != "cuda":
                logger.warning("⚠️ Diffusion disabled: CUDA not available (CPU-only system)")
                logger.info("💡 Structure extraction will work, but image generation requires a GPU")
                self.loaded = False
                self.pipeline = None
                return
            
            # Load the image-to-image pipeline (GPU only)
            logger.info(f"Loading diffusion model: {self.model_name} on {self.device}")
            
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                safety_checker=None,  # Disable safety checker for research
                requires_safety_checker=False
            )
            
            # Move to GPU
            self.pipeline = self.pipeline.to("cuda")
            
            # Enable memory efficient attention
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            self.loaded = True
            logger.info(f"✅ Diffusion model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load diffusion model: {str(e)}")
            self.loaded = False
            self.pipeline = None
    
    def encode_text(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        """
        Encode text prompt for conditioning.
        
        Args:
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt for guidance
            
        Returns:
            Text embeddings tensor
        """
        try:
            # Use the pipeline's built-in text encoding
            # This is more efficient than manual encoding
            return prompt, negative_prompt
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise
    
    def generate_edit(
        self,
        input_image_path: str,
        instruction: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        output_path: Optional[str] = None,
        negative_prompt: str = "blurry, low quality, distorted, deformed"
    ) -> Image.Image:
        """
        Generate edited image using diffusion model.
        
        Args:
            input_image_path: Path to input image
            instruction: Text instruction for editing
            strength: How much to transform the image (0-1)
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of denoising steps
            output_path: Path to save output image
            negative_prompt: Negative prompt for guidance
            
        Returns:
            Generated PIL Image
        """
        try:
            # Check if pipeline is loaded
            if self.pipeline is None:
                raise RuntimeError("Diffusion model not loaded. Check logs for details.")
            
            logger.info(f"Generating edit: '{instruction}' with strength {strength} on {self.device}")
            
            # Use CPU-optimized parameters if running on CPU
            if self.device == "cpu":
                num_inference_steps = min(num_inference_steps, CPU_INFERENCE_STEPS)
                guidance_scale = min(guidance_scale, CPU_GUIDANCE_SCALE)
                logger.info(f"🐌 CPU mode: Using {num_inference_steps} steps, guidance {guidance_scale}")
            
            # Load and preprocess input image
            input_image = Image.open(input_image_path).convert("RGB")
            
            # Resize if necessary (diffusion models work best with specific sizes)
            target_size = (512, 512)  # Standard size for Stable Diffusion
            if input_image.size != target_size:
                # Maintain aspect ratio
                input_image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create square canvas
                canvas = Image.new("RGB", target_size, (255, 255, 255))
                offset = ((target_size[0] - input_image.size[0]) // 2,
                         (target_size[1] - input_image.size[1]) // 2)
                canvas.paste(input_image, offset)
                input_image = canvas
            
            # Generate the edited image
            if self.device == "cpu":
                # CPU inference without autocast
                result = self.pipeline(
                    prompt=instruction,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                    generator=torch.Generator(device=self.device).manual_seed(42)  # For reproducibility
                )
            else:
                # GPU inference with autocast for speed
                with torch.autocast(self.device):
                    result = self.pipeline(
                        prompt=instruction,
                        image=input_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        negative_prompt=negative_prompt,
                        generator=torch.Generator(device=self.device).manual_seed(42)  # For reproducibility
                    )
            
            output_image = result.images[0]
            
            # Save if path provided
            if output_path:
                output_image.save(output_path)
                logger.info(f"Generated image saved to: {output_path}")
            
            logger.info("✅ Image generation completed successfully")
            return output_image
            
        except Exception as e:
            logger.error(f"❌ Failed to generate edit: {str(e)}")
            raise
    
    def generate_with_structural_constraints(
        self,
        input_image_path: str,
        instruction: str,
        structural_mask: Optional[np.ndarray] = None,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate edited image with structural constraints.
        
        Args:
            input_image_path: Path to input image
            instruction: Text instruction for editing
            structural_mask: Mask indicating regions to preserve/modify
            strength: How much to transform the image
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of denoising steps
            output_path: Path to save output image
            
        Returns:
            Generated PIL Image
        """
        try:
            # For now, use standard generation
            # Structural constraints will be applied in the controller module
            return self.generate_edit(
                input_image_path=input_image_path,
                instruction=instruction,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_path=output_path
            )
            
        except Exception as e:
            logger.error(f"Failed to generate with constraints: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "pipeline_type": "StableDiffusionImg2ImgPipeline",
            "is_loaded": self.pipeline is not None,
            "torch_dtype": str(next(self.pipeline.unet.parameters()).dtype) if self.pipeline else None
        }
    
    def unload_model(self):
        """Unload the model to free memory."""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
                torch.cuda.empty_cache() if self.device == "cuda" else None
                logger.info("Diffusion model unloaded from memory")
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()

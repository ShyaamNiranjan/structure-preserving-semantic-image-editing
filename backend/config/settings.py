import os
from pathlib import Path
from typing import Optional

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, INPUTS_DIR, OUTPUTS_DIR, INTERMEDIATE_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# ML Model Configuration
# Auto-detect CUDA availability
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available() and not FORCE_CPU
except ImportError:
    CUDA_AVAILABLE = False

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
MODEL_NAME = os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
INFERENCE_STEPS = int(os.getenv("INFERENCE_STEPS", "20"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))

# CPU-specific settings
CPU_INFERENCE_STEPS = int(os.getenv("CPU_INFERENCE_STEPS", "10"))  # Fewer steps for CPU
CPU_GUIDANCE_SCALE = float(os.getenv("CPU_GUIDANCE_SCALE", "5.0"))  # Lower guidance for CPU

# Diffusion Configuration
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "20"))
STRENGTH = float(os.getenv("STRENGTH", "0.8"))
ETA = float(os.getenv("ETA", "0.0"))

# Evaluation Metrics
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
COMPUTE_SSIM = os.getenv("COMPUTE_SSIM", "true").lower() == "true"
COMPUTE_LPIPS = os.getenv("COMPUTE_LPIPS", "true").lower() == "true"

# Storage Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/metadata.db")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Session Management
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "120"))
MAX_EDITS_PER_SESSION = int(os.getenv("MAX_EDITS_PER_SESSION", "50"))

# GPU Configuration
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
TORCH_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", None)

# Development Settings
ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

print(f"Configuration loaded. Device: {DEVICE}, CUDA Available: {CUDA_AVAILABLE}, Model: {MODEL_NAME}")

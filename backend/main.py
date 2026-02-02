from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path

from config.settings import API_HOST, API_PORT, API_DEBUG, ENABLE_CORS, CORS_ORIGINS, PROJECT_ROOT
from routes import upload, edit, image, history, evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Structure-Preserving Image Editing API",
    description="AI-powered image editing with structural constraints",
    version="1.0.0"
)

# Add CORS middleware if enabled
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files for frontend
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Include routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(edit.router, prefix="/api", tags=["edit"])
app.include_router(image.router, prefix="/api", tags=["image"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(evaluate.router, prefix="/api", tags=["evaluate"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Structure-Preserving Image Editing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/status")
async def api_status():
    """API status endpoint with device information"""
    try:
        from config.settings import DEVICE, CUDA_AVAILABLE, MODEL_NAME
        
        return {
            "status": "running",
            "device": DEVICE,
            "cuda_available": CUDA_AVAILABLE,
            "model_name": MODEL_NAME,
            "version": "1.0.0",
            "endpoints": {
                "upload": "/api/upload",
                "edit": "/api/edit",
                "image": "/api/image/{image_id}",
                "history": "/api/history",
                "evaluate": "/api/evaluate"
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "device": "unknown"
        }

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Structure-Preserving Image Editing API")
    
    # Initialize ML models (will be implemented in ML modules)
    try:
        # This will be implemented when we create the ML modules
        logger.info("ML models will be initialized on first use")
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Structure-Preserving Image Editing API")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
        log_level="info"
    )

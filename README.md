# Structure-Preserving Semantic Image Editing System

A domain-agnostic, structure-preserving generative framework built on latent diffusion models for controlled semantic image augmentation.

## 🎯 Overview

This system integrates image-to-image diffusion, semantic and structural perception, and incremental editing logic to modify real images based on user intent while preserving spatial layout, object boundaries, and visual realism through perceptual constraints and quantitative quality metrics.

## 🏗️ Architecture

### Backend (Python + FastAPI)
- **ML Inference Engine**: Stable Diffusion image-to-image generation
- **Structural Perception Module**: Edge detection, segmentation, semantic analysis
- **Structure-Preservation Controller**: Constraint application and incremental editing
- **Evaluation Module**: Quality metrics (SSIM, LPIPS, PSNR, edge preservation)
- **Storage Layer**: Image management and metadata database

### Frontend (HTML/CSS/JavaScript)
- Image upload and management
- Text-based edit instructions
- Real-time parameter adjustment
- Before/after comparison
- Quality metrics display
- Edit history tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- **For GPU**: CUDA-enabled GPU (6-8 GB VRAM minimum)
- **For CPU**: Any modern CPU (slower but functional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Paper Project"
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start the backend server**
```bash
cd backend
python main.py
```

6. **Open the frontend**
- Open `frontend/index.html` in your web browser
- Or serve with a local web server for better CORS handling

### 🖥️ CPU-Only Setup

If you're using a CPU-only machine, the system will automatically detect and optimize for CPU:

1. **Install CPU-optimized PyTorch** (if needed):
```bash
# For CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. **Set CPU mode** (optional, auto-detected):
```bash
# Create .env file
echo "FORCE_CPU=true" > .env
```

3. **Expected Performance**:
- ⏱️ **CPU**: 2-5 minutes per edit (slower but functional)
- ⚡ **GPU**: 5-15 seconds per edit (recommended)

The system will automatically:
- Use fewer inference steps on CPU (10 vs 20)
- Reduce guidance scale for faster processing
- Enable CPU memory optimizations
- Fall back to CPU if CUDA is unavailable

## 📁 Project Structure

```
Paper Project/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config/
│   │   └── settings.py         # Configuration management
│   ├── routes/                 # API endpoints
│   │   ├── upload.py
│   │   ├── edit.py
│   │   ├── image.py
│   │   ├── history.py
│   │   └── evaluate.py
│   ├── ml_modules/             # Machine learning components
│   │   ├── inference/
│   │   │   └── diffusion_engine.py
│   │   ├── perception/
│   │   │   └── structure_extractor.py
│   │   ├── controller/
│   │   │   └── structure_controller.py
│   │   └── evaluation/
│   │       └── metrics_calculator.py
│   └── storage/
│       └── image_manager.py
├── frontend/
│   ├── index.html              # Main web interface
│   ├── styles.css              # Styling
│   └── script.js               # Frontend logic
├── data/                       # Image storage
│   ├── inputs/
│   ├── outputs/
│   └── intermediate/
├── models/                     # Model checkpoints
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🔧 Configuration

Key environment variables (`.env` file):

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# ML Model Configuration
MODEL_NAME=runwayml/stable-diffusion-v1-5
FORCE_CPU=false
INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5

# Storage
MAX_FILE_SIZE_MB=50
DATABASE_URL=sqlite:///metadata.db

# GPU
CUDA_VISIBLE_DEVICES=0
```

## 🎨 Usage

### Basic Workflow

1. **Upload an Image**
   - Drag and drop or click to upload
   - Supported formats: PNG, JPG, BMP, TIFF
   - Maximum file size: 50MB

2. **Enter Edit Instruction**
   - Describe the changes you want to make
   - Example: "Make the sky more dramatic" or "Add flowers to the garden"

3. **Adjust Parameters**
   - **Strength**: How much to transform the image (0.1-1.0)
   - **Guidance Scale**: CFG guidance strength (1.0-15.0)
   - **Inference Steps**: Number of denoising steps (10-50)

4. **Apply Edit**
   - Click "Apply Edit" to generate the edited image
   - View results in the before/after comparison

5. **Review Metrics**
   - SSIM: Structural similarity
   - LPIPS: Perceptual similarity
   - PSNR: Peak signal-to-noise ratio
   - Edge Preservation: Structural edge maintenance

6. **Incremental Editing**
   - Each edit becomes the input for the next
   - Build complex edits step by step
   - Track history in the edit timeline

### API Usage

The system provides a RESTful API for programmatic access:

```python
import requests

# Upload image
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload', files={'file': f})
    image_id = response.json()['image_id']

# Apply edit
edit_request = {
    'image_id': image_id,
    'instruction_text': 'Make the image brighter',
    'strength': 0.8,
    'guidance_scale': 7.5,
    'num_inference_steps': 20
}
response = requests.post('http://localhost:8000/api/edit', json=edit_request)
result = response.json()
```

## 🧠 Model Details

### Diffusion Model
- **Base Model**: Stable Diffusion v1.5
- **Type**: Image-to-Image latent diffusion
- **Conditioning**: Text prompts + input image
- **Resolution**: 512x512 (standard)

### Structural Analysis
- **Edge Detection**: Canny, Sobel, Laplacian
- **Segmentation**: Felzenszwalb, SLIC superpixels
- **Semantic Analysis**: Optional DETR-based segmentation
- **Feature Extraction**: LBP, Gabor filters

### Quality Metrics
- **SSIM**: Structural similarity index
- **LPIPS**: Learned perceptual image patch similarity
- **PSNR**: Peak signal-to-noise ratio
- **Edge Preservation**: Edge overlap metrics
- **Semantic Consistency**: Feature-based similarity

## 📊 Performance

### Hardware Requirements
- **GPU**: CUDA-enabled, 6-8 GB VRAM minimum (recommended)
- **CPU**: Multi-core processor (supported but slower)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data

### Benchmarks
- **GPU Inference Time**: 5-15 seconds per edit (depending on parameters)
- **CPU Inference Time**: 2-5 minutes per edit (optimized but slower)
- **Memory Usage**: 4-6 GB GPU memory or 8GB+ RAM for CPU
- **Quality**: High structural preservation (>0.8 SSIM typical)

## 🔬 Research Applications

This system is designed for research in:
- **Structure-preserving image generation**
- **Controlled image augmentation**
- **Incremental editing methodologies**
- **Quality assessment metrics**
- **User-guided image synthesis**

## 🛠️ Development

### Adding New Metrics
```python
# In ml_modules/evaluation/metrics_calculator.py
def _calculate_custom_metric(self, img1, img2):
    # Your metric calculation
    return metric_value
```

### Extending Structural Analysis
```python
# In ml_modules/perception/structure_extractor.py
def _extract_custom_features(self, image):
    # Your feature extraction
    return features
```

### Custom Constraints
```python
# In ml_modules/controller/structure_controller.py
def _apply_custom_constraints(self, image, structural_info):
    # Your constraint logic
    return constrained_image
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image size or batch size
   - Use CPU mode: `FORCE_CPU=true`

2. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face access
   - Use alternative model in `.env`

3. **Slow Performance on CPU**
   - Reduce inference steps: `CPU_INFERENCE_STEPS=5`
   - Lower guidance scale: `CPU_GUIDANCE_SCALE=3.0`
   - Ensure CPU mode is enabled: `FORCE_CPU=true`

4. **"Torch not compiled with CUDA enabled"**
   - System will automatically fall back to CPU
   - Or install CPU-only PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

5. **API Connection Issues**
   - Check backend is running on port 8000
   - Verify CORS settings
   - Check firewall settings

### Logs
- Application logs: `logs/`
- Model loading: Check console output
- API errors: Check browser console

## 📝 License

This project is for research purposes. Please refer to the license file for usage terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- Diffusers Library: https://github.com/huggingface/diffusers
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- OpenCV: https://opencv.org/
- Scikit-image: https://scikit-image.org/

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Open an issue on the repository

---

**Note**: This is a research prototype. Performance and features are continuously being improved.

# GitHub Upload Script for Structure-Preserving Image Editing
# Run this in PowerShell as Administrator

Write-Host "🚀 Starting GitHub Upload Process..." -ForegroundColor Green

# Navigate to project directory
Set-Location "C:\Users\shyaa\CascadeProjects\Paper Project"
Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Blue

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "🔧 Initializing Git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "✅ Git repository already initialized" -ForegroundColor Green
}

# Add remote origin
Write-Host "🔗 Adding remote origin..." -ForegroundColor Yellow
git remote add origin https://github.com/ShyaamNiranjan/structure-preserving-image-editing.git

# Set up main branch
Write-Host "🌿 Setting up main branch..." -ForegroundColor Yellow
git branch -M main

# Add all files
Write-Host "📦 Adding all files to staging..." -ForegroundColor Yellow
git add .

# Check status
Write-Host "📋 Git status:" -ForegroundColor Blue
git status

# Commit with professional message
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m @"
Initial commit: Structure-preserving image editing system

🎯 Features:
- AI-powered image editing with structural constraints
- FastAPI backend with ML modules
- Modern glassmorphism frontend
- CPU/GPU adaptive processing
- Real-time quality metrics (SSIM, LPIPS, PSNR)
- Complete test suite and documentation

📚 Research:
- IEEE paper project for Data Science Masters
- Structure-preserving diffusion models
- Quantitative evaluation framework

🚀 Tech Stack:
- Backend: Python, FastAPI, PyTorch, OpenCV
- Frontend: HTML5, CSS3, JavaScript
- ML: Diffusion models, Computer vision
- Metrics: SSIM, LPIPS, PSNR, Structure preservation

👤 Author: Shyaam Niranjan
📧 Contact: For academic and research collaboration
"@

# Push to GitHub
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Yellow
try {
    git push -u origin main
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "🌐 Repository: https://github.com/ShyaamNiranjan/structure-preserving-image-editing" -ForegroundColor Blue
} catch {
    Write-Host "❌ Push failed. You may need to:" -ForegroundColor Red
    Write-Host "1. Create the repository on GitHub first" -ForegroundColor Yellow
    Write-Host "2. Check your GitHub credentials" -ForegroundColor Yellow
    Write-Host "3. Try running: git push -u origin main --force" -ForegroundColor Yellow
}

Write-Host "🎉 GitHub upload process completed!" -ForegroundColor Green
Write-Host "📊 Next steps:" -ForegroundColor Blue
Write-Host "1. Visit your repository on GitHub" -ForegroundColor White
Write-Host "2. Add README badges and screenshots" -ForegroundColor White
Write-Host "3. Create GitHub Pages for demo" -ForegroundColor White
Write-Host "4. Add to your CV and DAAD applications" -ForegroundColor White

# Keep window open
Read-Host "Press Enter to exit"

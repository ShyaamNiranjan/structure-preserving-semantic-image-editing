// Modern StructureAI Frontend Script

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';

class StructureAI {
    constructor() {
        this.currentImageId = null;
        this.currentSessionId = null;
        this.isProcessing = false;
        this.editHistory = [];
        this.comparisonMode = 'side-by-side';
        
        this.init();
    }

    init() {
        this.initializeElements();
        this.setupEventListeners();
        this.checkBackendStatus();
        this.loadTheme();
    }

    async checkBackendStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            if (response.ok) {
                this.showStatus('Connected to backend successfully!', 'success');
                this.createSession();
                this.checkDeviceMode();
            } else {
                throw new Error('Backend not responding correctly');
            }
        } catch (error) {
            console.error('Backend connection failed:', error);
            this.showStatus('❌ Backend not running! Please start the backend server first.', 'error');
            this.showBackendInstructions();
        }
    }

    showBackendInstructions() {
        const instructions = `
            <div style="text-align: left; margin-top: 1rem;">
                <strong>🚀 To start the backend:</strong><br>
                1. Open terminal<br>
                2. Navigate to: <code>cd backend</code><br>
                3. Run: <code>python main.py</code><br>
                4. Wait for "Uvicorn running on http://0.0.0.0:8000"<br>
                <br>
                <strong>💡 Then refresh this page</strong>
            </div>
        `;
        
        const statusEl = document.createElement('div');
        statusEl.className = 'status-message error';
        statusEl.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <div>
                <strong>Backend Connection Failed</strong>
                ${instructions}
            </div>
        `;
        
        this.statusContainer.appendChild(statusEl);
    }

    initializeElements() {
        // Navigation
        this.themeToggle = document.getElementById('themeToggle');
        this.themeIcon = document.getElementById('themeIcon');
        this.infoBtn = document.getElementById('infoBtn');
        this.infoModal = document.getElementById('infoModal');
        this.closeModal = document.getElementById('closeModal');
        
        // Upload
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.uploadInfo = document.getElementById('uploadInfo');
        this.previewImage = document.getElementById('previewImage');
        this.uploadedFileName = document.getElementById('uploadedFileName');
        this.fileSize = document.getElementById('fileSize');
        this.removeImageBtn = document.getElementById('removeImage');
        
        // Edit Controls
        this.instructionInput = document.getElementById('instructionInput');
        this.strengthSlider = document.getElementById('strengthSlider');
        this.strengthValue = document.getElementById('strengthValue');
        this.guidanceSlider = document.getElementById('guidanceSlider');
        this.guidanceValue = document.getElementById('guidanceValue');
        this.stepsSlider = document.getElementById('stepsSlider');
        this.stepsValue = document.getElementById('stepsValue');
        this.applyEditBtn = document.getElementById('applyEditBtn');
        
        // Images
        this.originalImage = document.getElementById('originalImage');
        this.editedImage = document.getElementById('editedImage');
        this.originalPlaceholder = document.getElementById('originalPlaceholder');
        this.editedPlaceholder = document.getElementById('editedPlaceholder');
        
        // Comparison
        this.comparisonBtns = document.querySelectorAll('.comparison-btn');
        this.imageComparison = document.getElementById('imageComparison');
        
        // Metrics
        this.ssimValue = document.getElementById('ssimValue');
        this.lpipsValue = document.getElementById('lpipsValue');
        this.psnrValue = document.getElementById('psnrValue');
        this.structureValue = document.getElementById('structureValue');
        
        // History
        this.historyTimeline = document.getElementById('historyTimeline');
        this.clearHistoryBtn = document.getElementById('clearHistory');
        
        // Status
        this.statusContainer = document.getElementById('statusContainer');
        
        // Device Badge
        this.deviceBadge = document.getElementById('deviceBadge');
    }

    setupEventListeners() {
        // Theme Toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
        
        // Info Modal
        this.infoBtn.addEventListener('click', () => this.showInfoModal());
        this.closeModal.addEventListener('click', () => this.hideInfoModal());
        this.infoModal.addEventListener('click', (e) => {
            if (e.target === this.infoModal) this.hideInfoModal();
        });
        
        // Upload
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.removeImageBtn.addEventListener('click', () => this.removeImage());
        
        // Edit Controls
        this.strengthSlider.addEventListener('input', (e) => this.updateSliderValue('strength', e.target.value));
        this.guidanceSlider.addEventListener('input', (e) => this.updateSliderValue('guidance', e.target.value));
        this.stepsSlider.addEventListener('input', (e) => this.updateSliderValue('steps', e.target.value));
        this.applyEditBtn.addEventListener('click', () => this.applyEdit());
        
        // Instruction Input
        this.instructionInput.addEventListener('input', () => this.validateForm());
        
        // Suggestion Chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                this.instructionInput.value = chip.dataset.instruction;
                this.validateForm();
            });
        });
        
        // Comparison Controls
        this.comparisonBtns.forEach(btn => {
            btn.addEventListener('click', () => this.setComparisonMode(btn.dataset.view));
        });
        
        // Clear History
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    // Theme Management
    loadTheme() {
        const theme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        this.updateThemeIcon(theme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
        
        this.showStatus(`Theme changed to ${newTheme} mode`, 'success');
    }

    updateThemeIcon(theme) {
        this.themeIcon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }

    // Device Mode Check
    async checkDeviceMode() {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            const data = await response.json();
            
            if (data.device === 'cpu') {
                this.deviceBadge.innerHTML = '<i class="fas fa-microchip"></i> CPU Mode';
                this.deviceBadge.style.background = 'var(--warning)';
            } else {
                this.deviceBadge.innerHTML = '<i class="fas fa-rocket"></i> GPU Mode';
                this.deviceBadge.style.background = 'var(--success)';
            }
        } catch (error) {
            console.error('Failed to check device mode:', error);
        }
    }

    // Modal Management
    showInfoModal() {
        this.infoModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    hideInfoModal() {
        this.infoModal.style.display = 'none';
        document.body.style.overflow = '';
    }

    // File Upload
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        // Validate file
        if (!this.validateFile(file)) return;
        
        // Show progress
        this.showUploadProgress();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.handleUploadSuccess(result, file);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus(`Upload failed: ${error.message}`, 'error');
            this.hideUploadProgress();
        }
    }

    validateFile(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        const maxSize = 50 * 1024 * 1024; // 50MB
        
        if (!allowedTypes.includes(file.type)) {
            this.showStatus('Invalid file type. Please upload an image.', 'error');
            return false;
        }
        
        if (file.size > maxSize) {
            this.showStatus('File too large. Maximum size is 50MB.', 'error');
            return false;
        }
        
        return true;
    }

    showUploadProgress() {
        this.uploadProgress.style.display = 'block';
        this.simulateProgress();
    }

    simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;
            
            this.progressFill.style.width = `${progress}%`;
            this.progressText.textContent = `Uploading... ${Math.round(progress)}%`;
            
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 200);
    }

    hideUploadProgress() {
        this.uploadProgress.style.display = 'none';
        this.progressFill.style.width = '0%';
    }

    handleUploadSuccess(result, file) {
        this.hideUploadProgress();
        
        // Update UI
        this.currentImageId = result.image_id;
        this.uploadedFileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.previewImage.src = `${API_BASE_URL}/image/${result.image_id}`;
        
        // Show uploaded state
        this.uploadArea.style.display = 'none';
        this.uploadInfo.style.display = 'flex';
        
        // Display original image
        this.displayOriginalImage(result.image_id);
        
        // Validate form
        this.validateForm();
        
        this.showStatus('Image uploaded successfully!', 'success');
    }

    removeImage() {
        this.currentImageId = null;
        
        // Reset upload UI
        this.uploadArea.style.display = 'block';
        this.uploadInfo.style.display = 'none';
        this.imageInput.value = '';
        
        // Reset image display
        this.originalImage.style.display = 'none';
        this.originalPlaceholder.style.display = 'flex';
        this.editedImage.style.display = 'none';
        this.editedPlaceholder.style.display = 'flex';
        
        // Reset metrics
        this.resetMetrics();
        
        // Validate form
        this.validateForm();
        
        this.showStatus('Image removed', 'success');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Image Display
    displayOriginalImage(imageId) {
        this.originalImage.src = `${API_BASE_URL}/image/${imageId}`;
        this.originalImage.style.display = 'block';
        this.originalPlaceholder.style.display = 'none';
        
        // Setup download button
        document.getElementById('downloadOriginal').style.display = 'block';
        document.getElementById('downloadOriginal').onclick = () => {
            this.downloadImage(imageId, 'original');
        };
    }

    displayEditedImage(imageId) {
        this.editedImage.src = `${API_BASE_URL}/image/${imageId}`;
        this.editedImage.style.display = 'block';
        this.editedPlaceholder.style.display = 'none';
        
        // Setup download button
        document.getElementById('downloadEdited').style.display = 'block';
        document.getElementById('downloadEdited').onclick = () => {
            this.downloadImage(imageId, 'edited');
        };
        
        // Update current image for incremental editing
        this.currentImageId = imageId;
    }

    downloadImage(imageId, prefix) {
        const link = document.createElement('a');
        link.href = `${API_BASE_URL}/image/${imageId}`;
        link.download = `${prefix}_${imageId}.png`;
        link.click();
    }

    // Edit Controls
    updateSliderValue(type, value) {
        switch (type) {
            case 'strength':
                this.strengthValue.textContent = value;
                break;
            case 'guidance':
                this.guidanceValue.textContent = value;
                break;
            case 'steps':
                this.stepsValue.textContent = value;
                break;
        }
    }

    validateForm() {
        const hasImage = this.currentImageId !== null;
        const hasInstruction = this.instructionInput.value.trim().length > 0;
        
        this.applyEditBtn.disabled = !hasImage || !hasInstruction || this.isProcessing;
    }

    async applyEdit() {
        if (!this.currentImageId || !this.currentSessionId || this.isProcessing) return;
        
        try {
            this.isProcessing = true;
            this.updateEditButton();
            this.showLoading(true);
            
            const editRequest = {
                image_id: this.currentImageId,
                instruction_text: this.instructionInput.value.trim(),
                strength: parseFloat(this.strengthSlider.value),
                guidance_scale: parseFloat(this.guidanceSlider.value),
                num_inference_steps: parseInt(this.stepsSlider.value)
            };
            
            const response = await fetch(`${API_BASE_URL}/edit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(editRequest)
            });
            
            if (!response.ok) {
                throw new Error(`Edit failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.handleEditResponse(result);
            
        } catch (error) {
            console.error('Edit error:', error);
            this.showStatus(`Edit failed: ${error.message}`, 'error');
        } finally {
            this.isProcessing = false;
            this.updateEditButton();
            this.showLoading(false);
        }
    }

    handleEditResponse(result) {
        if (result.status === 'partial_success') {
            this.showStatus(result.message, 'warning');
            if (result.metrics) {
                this.updateMetrics(result.metrics);
            }
            return;
        }
        
        // Display edited image
        this.displayEditedImage(result.output_image_id);
        
        // Update metrics
        this.updateMetrics(result.metrics);
        
        // Add to history
        this.addToHistory(result);
        
        // Show success message
        this.showStatus('Edit applied successfully!', 'success');
    }

    updateEditButton() {
        const btnContent = this.applyEditBtn.querySelector('.btn-content');
        const btnLoading = this.applyEditBtn.querySelector('.btn-loading');
        
        if (this.isProcessing) {
            btnContent.style.display = 'none';
            btnLoading.style.display = 'flex';
        } else {
            btnContent.style.display = 'flex';
            btnLoading.style.display = 'none';
        }
    }

    showLoading(show) {
        if (show) {
            this.editedPlaceholder.innerHTML = `
                <div class="loading-spinner"></div>
                <span>Processing your edit...</span>
            `;
        } else {
            this.editedPlaceholder.innerHTML = `
                <i class="fas fa-magic"></i>
                <span>Edited image will appear here</span>
            `;
        }
    }

    // Comparison Mode
    setComparisonMode(mode) {
        this.comparisonMode = mode;
        
        // Update button states
        this.comparisonBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === mode);
        });
        
        // Update display (simplified for now)
        if (mode === 'slider') {
            this.showStatus('Slider comparison coming soon!', 'info');
        } else if (mode === 'diff') {
            this.showStatus('Difference view coming soon!', 'info');
        }
    }

    // Metrics
    updateMetrics(metrics) {
        if (metrics.ssim !== undefined) {
            this.ssimValue.textContent = metrics.ssim.toFixed(4);
        }
        if (metrics.lpips !== undefined) {
            this.lpipsValue.textContent = metrics.lpips.toFixed(4);
        }
        if (metrics.psnr !== undefined) {
            this.psnrValue.textContent = metrics.psnr.toFixed(2);
        }
        if (metrics.structure !== undefined) {
            this.structureValue.textContent = metrics.structure.toFixed(4);
        }
        
        // Handle CPU mode metrics
        if (metrics.diffusion_available !== undefined) {
            this.structureValue.textContent = metrics.diffusion_available ? 'Available' : 'N/A';
        }
    }

    resetMetrics() {
        this.ssimValue.textContent = '--';
        this.lpipsValue.textContent = '--';
        this.psnrValue.textContent = '--';
        this.structureValue.textContent = '--';
    }

    // History
    addToHistory(result) {
        const historyItem = {
            id: Date.now(),
            instruction: this.instructionInput.value.trim(),
            output_image_id: result.output_image_id,
            metrics: result.metrics,
            timestamp: new Date().toISOString()
        };
        
        this.editHistory.unshift(historyItem);
        this.renderHistory();
    }

    renderHistory() {
        if (this.editHistory.length === 0) {
            this.historyTimeline.innerHTML = `
                <div class="history-empty">
                    <i class="fas fa-clock"></i>
                    <span>No edits yet</span>
                </div>
            `;
            return;
        }
        
        const historyHTML = this.editHistory.map(item => `
            <div class="history-item">
                <div class="history-content">
                    <div class="history-instruction">${item.instruction}</div>
                    <div class="history-time">${this.formatTime(item.timestamp)}</div>
                </div>
                <div class="history-actions">
                    <button class="history-restore" onclick="app.restoreFromHistory('${item.output_image_id}')">
                        <i class="fas fa-undo"></i>
                    </button>
                </div>
            </div>
        `).join('');
        
        this.historyTimeline.innerHTML = historyHTML;
    }

    clearHistory() {
        this.editHistory = [];
        this.renderHistory();
        this.showStatus('History cleared', 'success');
    }

    restoreFromHistory(imageId) {
        this.displayEditedImage(imageId);
        this.showStatus('Restored from history', 'success');
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    // Session Management
    async createSession() {
        try {
            const response = await fetch(`${API_BASE_URL}/session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                throw new Error('Failed to create session');
            }
            
            const result = await response.json();
            this.currentSessionId = result.session_id;
            
        } catch (error) {
            console.error('Session creation failed:', error);
            this.showStatus('Failed to create session', 'error');
        }
    }

    // Status Messages
    showStatus(message, type = 'info') {
        const statusEl = document.createElement('div');
        statusEl.className = `status-message ${type}`;
        
        const icon = this.getStatusIcon(type);
        statusEl.innerHTML = `<i class="fas ${icon}"></i><span>${message}</span>`;
        
        this.statusContainer.appendChild(statusEl);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            statusEl.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => statusEl.remove(), 300);
        }, 5000);
    }

    getStatusIcon(type) {
        switch (type) {
            case 'success': return 'fa-check-circle';
            case 'error': return 'fa-exclamation-circle';
            case 'warning': return 'fa-exclamation-triangle';
            default: return 'fa-info-circle';
        }
    }

    // Keyboard Shortcuts
    handleKeyboard(e) {
        // Ctrl/Cmd + Enter to apply edit
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (!this.applyEditBtn.disabled) {
                this.applyEdit();
            }
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            this.hideInfoModal();
        }
        
        // Ctrl/Cmd + O to open file
        if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
            e.preventDefault();
            this.imageInput.click();
        }
    }
}

// Initialize app
const app = new StructureAI();

// Add slide out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        to {
            transform: translateX(120%);
            opacity: 0;
        }
    }
    
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--spacing-md);
        background: var(--bg-secondary);
        border-radius: var(--radius-lg);
        margin-bottom: var(--spacing-sm);
        border: 1px solid var(--border-color);
        transition: all var(--transition-normal);
    }
    
    .history-item:hover {
        transform: translateX(4px);
        box-shadow: var(--glass-shadow);
    }
    
    .history-content {
        flex: 1;
    }
    
    .history-instruction {
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
    }
    
    .history-time {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .history-actions {
        display: flex;
        gap: var(--spacing-sm);
    }
    
    .history-restore {
        width: 32px;
        height: 32px;
        border: none;
        background: var(--primary);
        color: white;
        border-radius: var(--radius-md);
        cursor: pointer;
        transition: all var(--transition-normal);
    }
    
    .history-restore:hover {
        background: var(--primary-dark);
        transform: scale(1.1);
    }
`;
document.head.appendChild(style);

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';

// Application State
let currentImageId = null;
let currentSessionId = null;
let editHistory = [];
let isProcessing = false;

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    imageInput: document.getElementById('imageInput'),
    uploadInfo: document.getElementById('uploadInfo'),
    uploadedFileName: document.getElementById('uploadedFileName'),
    removeImageBtn: document.getElementById('removeImage'),
    instructionInput: document.getElementById('instructionInput'),
    strengthSlider: document.getElementById('strengthSlider'),
    strengthValue: document.getElementById('strengthValue'),
    guidanceSlider: document.getElementById('guidanceSlider'),
    guidanceValue: document.getElementById('guidanceValue'),
    stepsSlider: document.getElementById('stepsSlider'),
    stepsValue: document.getElementById('stepsValue'),
    applyEditBtn: document.getElementById('applyEditBtn'),
    originalImageWrapper: document.getElementById('originalImageWrapper'),
    editedImageWrapper: document.getElementById('editedImageWrapper'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    metricsGrid: document.getElementById('metricsGrid'),
    historyList: document.getElementById('historyList'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    statusContainer: document.getElementById('statusContainer')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadSessionHistory();
    checkAPIHealth();
});

// Event Listeners
function initializeEventListeners() {
    // Upload events
    elements.uploadArea.addEventListener('click', () => elements.imageInput.click());
    elements.imageInput.addEventListener('change', handleImageUpload);
    elements.removeImageBtn.addEventListener('click', removeImage);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Edit controls
    elements.instructionInput.addEventListener('input', updateEditButton);
    elements.strengthSlider.addEventListener('input', updateSliderValue);
    elements.guidanceSlider.addEventListener('input', updateSliderValue);
    elements.stepsSlider.addEventListener('input', updateSliderValue);
    elements.applyEditBtn.addEventListener('click', applyEdit);
    
    // History controls
    elements.newSessionBtn.addEventListener('click', createNewSession);
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
}

// Image Upload Functions
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        uploadImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        uploadImage(file);
    } else {
        showStatus('Please drop a valid image file', 'error');
    }
}

async function uploadImage(file) {
    try {
        showLoading(true);
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const result = await response.json();
        currentImageId = result.image_id;
        
        // Create new session
        currentSessionId = await createSession();
        
        // Update UI
        displayUploadedImage(result);
        updateEditButton();
        showStatus('Image uploaded successfully', 'success');
        
    } catch (error) {
        console.error('Upload error:', error);
        showStatus(`Upload failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function displayUploadedImage(imageInfo) {
    elements.uploadArea.style.display = 'none';
    elements.uploadInfo.style.display = 'flex';
    elements.uploadedFileName.textContent = imageInfo.filename;
    
    // Display original image
    const img = document.createElement('img');
    img.src = `${API_BASE_URL}/image/${imageInfo.image_id}`;
    img.alt = 'Uploaded image';
    
    elements.originalImageWrapper.innerHTML = '';
    elements.originalImageWrapper.appendChild(img);
}

function removeImage() {
    currentImageId = null;
    elements.uploadArea.style.display = 'block';
    elements.uploadInfo.style.display = 'none';
    elements.imageInput.value = '';
    elements.originalImageWrapper.innerHTML = `
        <div class="no-image-placeholder">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
            <p>No image uploaded</p>
        </div>
    `;
    updateEditButton();
}

// Edit Functions
function updateEditButton() {
    const hasImage = currentImageId !== null;
    const hasInstruction = elements.instructionInput.value.trim().length > 0;
    elements.applyEditBtn.disabled = !(hasImage && hasInstruction) || isProcessing;
}

function updateSliderValue(event) {
    const slider = event.target;
    const valueSpan = document.getElementById(slider.id.replace('Slider', 'Value'));
    valueSpan.textContent = slider.value;
}

async function applyEdit() {
    if (!currentImageId || !currentSessionId || isProcessing) return;
    
    try {
        isProcessing = true;
        updateEditButton();
        showLoading(true);
        
        const editRequest = {
            image_id: currentImageId,
            instruction_text: elements.instructionInput.value.trim(),
            step_id: editHistory.length,
            strength: parseFloat(elements.strengthSlider.value),
            guidance_scale: parseFloat(elements.guidanceSlider.value),
            num_inference_steps: parseInt(elements.stepsSlider.value)
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
        
        // Handle different response statuses
        if (result.status === 'partial_success') {
            showStatus(result.message, 'warning');
            if (result.metrics) {
                updateMetrics(result.metrics);
            }
            return;
        }
        
        // Update current image for incremental editing
        currentImageId = result.output_image_id;
        
        // Display edited image
        displayEditedImage(result.output_image_id);
        
        // Update metrics
        updateMetrics(result.metrics);
        
        // Add to history
        addToHistory(editRequest, result);
        
        // Clear instruction
        elements.instructionInput.value = '';
        updateEditButton();
        
        showStatus('Edit applied successfully', 'success');
        
    } catch (error) {
        console.error('Edit error:', error);
        showStatus(`Edit failed: ${error.message}`, 'error');
    } finally {
        isProcessing = false;
        updateEditButton();
        showLoading(false);
    }
}

function displayEditedImage(imageId) {
    const img = document.createElement('img');
    img.src = `${API_BASE_URL}/image/${imageId}`;
    img.alt = 'Edited image';
    
    elements.editedImageWrapper.innerHTML = '';
    elements.editedImageWrapper.appendChild(img);
}

function updateMetrics(metrics) {
    const metricElements = {
        ssim: document.getElementById('ssimValue'),
        lpips: document.getElementById('lpipsValue'),
        psnr: document.getElementById('psnrValue'),
        edge_preservation: document.getElementById('edgeValue')
    };
    
    for (const [key, element] of Object.entries(metricElements)) {
        if (metrics[key] !== undefined) {
            const value = typeof metrics[key] === 'number' ? metrics[key].toFixed(4) : metrics[key];
            element.textContent = value;
        }
    }
}

// Session Management
async function createSession() {
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
        return result.session_id;
        
    } catch (error) {
        console.error('Session creation error:', error);
        // Fallback to generated session ID
        return 'session_' + Date.now();
    }
}

function createNewSession() {
    if (confirm('Create a new session? Current progress will be saved.')) {
        currentSessionId = null;
        editHistory = [];
        currentImageId = null;
        removeImage();
        elements.editedImageWrapper.innerHTML = `
            <div class="no-image-placeholder">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                <p>No edits applied</p>
            </div>
        `;
        resetMetrics();
        showStatus('New session created', 'info');
    }
}

// History Management
function addToHistory(editRequest, result) {
    const historyItem = {
        step_id: editRequest.step_id,
        instruction: editRequest.instruction_text,
        output_image_id: result.output_image_id,
        metrics: result.metrics,
        processing_time: result.processing_time,
        timestamp: new Date().toISOString()
    };
    
    editHistory.push(historyItem);
    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    if (editHistory.length === 0) {
        elements.historyList.innerHTML = `
            <div class="no-history">
                <p>No edit history yet. Upload an image and start editing!</p>
            </div>
        `;
        return;
    }
    
    const historyHTML = editHistory.map((item, index) => `
        <div class="history-item">
            <img src="${API_BASE_URL}/image/${item.output_image_id}" alt="Edit step ${index + 1}">
            <div class="history-info">
                <div class="history-step">Step ${item.step_id + 1}</div>
                <div class="history-instruction">${item.instruction}</div>
                <div class="history-metrics">
                    SSIM: ${item.metrics.ssim?.toFixed(3) || '-'} | 
                    LPIPS: ${item.metrics.lpips?.toFixed(3) || '-'} | 
                    Time: ${item.processing_time?.toFixed(1) || '-'}s
                </div>
            </div>
        </div>
    `).join('');
    
    elements.historyList.innerHTML = historyHTML;
}

async function loadSessionHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history`);
        if (response.ok) {
            const data = await response.json();
            // Display available sessions if needed
        }
    } catch (error) {
        console.error('Failed to load session history:', error);
    }
}

function clearHistory() {
    if (confirm('Clear all edit history? This cannot be undone.')) {
        editHistory = [];
        updateHistoryDisplay();
        showStatus('History cleared', 'info');
    }
}

// Utility Functions
function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showStatus(message, type = 'info') {
    const statusElement = document.createElement('div');
    statusElement.className = `status-message ${type}`;
    statusElement.textContent = message;
    
    elements.statusContainer.appendChild(statusElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (statusElement.parentNode) {
            statusElement.parentNode.removeChild(statusElement);
        }
    }, 5000);
}

function resetMetrics() {
    document.getElementById('ssimValue').textContent = '-';
    document.getElementById('lpipsValue').textContent = '-';
    document.getElementById('psnrValue').textContent = '-';
    document.getElementById('edgeValue').textContent = '-';
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL.replace('/api', '')}/health`);
        if (response.ok) {
            showStatus('API server is running', 'success');
        } else {
            showStatus('API server may not be running', 'warning');
        }
    } catch (error) {
        showStatus('Cannot connect to API server', 'error');
    }
}

// Error Handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showStatus('An unexpected error occurred', 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showStatus('An unexpected error occurred', 'error');
});

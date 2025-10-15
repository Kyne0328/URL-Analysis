// Menu toggle functionality
document.getElementById('menuToggle').addEventListener('click', function() {
    this.classList.toggle('active');
    document.getElementById('mainMenu').classList.toggle('active');
});

// Smooth scrolling and active menu update
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
            // Close mobile menu if open
            document.getElementById('menuToggle').classList.remove('active');
            document.getElementById('mainMenu').classList.remove('active');
            
            // Update active menu item immediately
            updateActiveMenuItem();
        }
    });
});

// Update active menu item based on scroll position
function updateActiveMenuItem() {
    const sections = document.querySelectorAll('.section');
    const menuItems = document.querySelectorAll('.menu-item:not(.external)');
    const scrollPosition = window.scrollY + 100;
    
    sections.forEach((section, index) => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            menuItems.forEach(item => item.classList.remove('active'));
            if (menuItems[index]) {
                menuItems[index].classList.add('active');
            }
        }
    });
}

// Header scroll effect
window.addEventListener('scroll', function() {
    const header = document.getElementById('header');
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
    
    // Update active menu item on scroll
    updateActiveMenuItem();
});

// Scroll to top button
const scrollToTopBtn = document.getElementById('scrollToTop');
window.addEventListener('scroll', function() {
    if (window.scrollY > 300) {
        scrollToTopBtn.classList.add('visible');
    } else {
        scrollToTopBtn.classList.remove('visible');
    }
});

scrollToTopBtn.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Logo click to scroll to home
document.querySelector('.logo-container').addEventListener('click', function(e) {
    e.preventDefault();
    const homeSection = document.getElementById('home');
    if (homeSection) {
        homeSection.scrollIntoView({ behavior: 'smooth' });
    }
});

let currentUrl = '';
let currentResults = null;

// Toast Notification System
function showToast(message, type = 'info', title = '') {
    const toastContainer = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        error: 'fa-times-circle',
        success: 'fa-check-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-triangle'
    };
    
    const titles = {
        error: title || 'Error',
        success: title || 'Success',
        info: title || 'Info',
        warning: title || 'Warning'
    };
    
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${icons[type]}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-title">${titles[type]}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// URL Validation
function validateURL(url) {
    try {
        const urlPattern = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
        return urlPattern.test(url);
    } catch (e) {
        return false;
    }
}

// Real-time URL validation with ARIA updates
const urlInput = document.getElementById('urlInput');
const validationIcon = document.getElementById('validationIcon');

urlInput.addEventListener('input', function() {
    const url = this.value.trim();
    
    if (url.length === 0) {
        this.classList.remove('valid', 'invalid');
        this.setAttribute('aria-invalid', 'false');
        validationIcon.classList.remove('show', 'valid', 'invalid');
        validationIcon.innerHTML = '';
        return;
    }
    
    if (validateURL(url)) {
        this.classList.remove('invalid');
        this.classList.add('valid');
        this.setAttribute('aria-invalid', 'false');
        validationIcon.classList.remove('invalid');
        validationIcon.classList.add('valid', 'show');
        validationIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
    } else {
        this.classList.remove('valid');
        this.classList.add('invalid');
        this.setAttribute('aria-invalid', 'true');
        validationIcon.classList.remove('valid');
        validationIcon.classList.add('invalid', 'show');
        validationIcon.innerHTML = '<i class="fas fa-times-circle"></i>';
    }
});

// Example URL buttons
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const url = this.getAttribute('data-url');
        urlInput.value = url;
        urlInput.dispatchEvent(new Event('input'));
        urlInput.focus();
        showToast('Example URL loaded. Click "Analyze" to proceed.', 'info', 'Example Loaded');
    });
});

// URL Analysis
document.getElementById('urlForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const url = document.getElementById('urlInput').value.trim();
    if (!url) {
        showToast('Please enter a URL to analyze', 'warning', 'Input Required');
        return;
    }
    
    if (!validateURL(url)) {
        showToast('Please enter a valid URL format (e.g., https://example.com)', 'error', 'Invalid URL');
        return;
    }
    
    currentUrl = url;
    showLoading(true);
    showToast('Starting URL pattern analysis...', 'info', 'Analyzing');
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentResults = data;
            displayResults(data);
            showResultsSection(true);
            showToast('Analysis complete! Check results below.', 'success', 'Complete');
            
            // Smooth scroll to results
            setTimeout(() => {
                document.getElementById('resultsSection').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 300);
        } else {
            showToast(data.error || 'Analysis failed. Please try again.', 'error', 'Analysis Failed');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error', 'Connection Error');
    } finally {
        showLoading(false);
    }
});

// Tooltip helper function
function createTooltip(label, content) {
    return `
        <div class="tooltip-wrapper">
            <span>${label}</span>
            <span class="tooltip-trigger" tabindex="0" role="tooltip" aria-label="${content}">
                <i class="fas fa-question-circle"></i>
                <span class="tooltip-content">${content}</span>
            </span>
        </div>
    `;
}

// Extract domain from URL
function extractDomain(url) {
    try {
        const urlObj = new URL(url.startsWith('http') ? url : 'https://' + url);
        return urlObj.hostname;
    } catch {
        return url.split('/')[0];
    }
}

// Format URL with domain highlighting
function formatURLDisplay(url) {
    try {
        const urlObj = new URL(url.startsWith('http') ? url : 'https://' + url);
        const domain = urlObj.hostname;
        const path = urlObj.pathname + urlObj.search + urlObj.hash;
        return `<span class="domain-part">${domain}</span><span class="path-part">${path}</span>`;
    } catch {
        return url;
    }
}

// Get favicon letter
function getFaviconLetter(domain) {
    const cleanDomain = domain.replace('www.', '');
    return cleanDomain.charAt(0).toUpperCase();
}

// Display Results
function displayResults(data) {
    const urlInfo = data.url_info;
    
    // Update result icon and title
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    
    let iconClass = 'fas fa-question-circle';
    let iconStyle = 'mixed';
    let title = 'Analysis Results';
    let subtitle = 'URL pattern analysis complete';
    
    // Determine result styling
    if (urlInfo.prediction === 'suspicious_pattern') {
        iconClass = 'fas fa-exclamation-triangle';
        iconStyle = 'suspicious';
        title = 'Suspicious Pattern Detected';
        subtitle = 'This URL matches suspicious pattern groups';
    } else if (urlInfo.prediction === 'normal_pattern') {
        iconClass = 'fas fa-check-circle';
        iconStyle = 'safe';
        title = 'Normal Pattern Detected';
        subtitle = 'This URL matches normal pattern groups';
    } else if (urlInfo.prediction === 'mixed_pattern') {
        iconClass = 'fas fa-question-circle';
        iconStyle = 'mixed';
        title = 'Mixed Pattern Group';
        subtitle = 'This cluster contains both legitimate and suspicious URLs';
    } else if (urlInfo.prediction === 'uncertain') {
        iconClass = 'fas fa-question-circle';
        iconStyle = 'mixed';
        title = 'Uncertain Pattern';
        subtitle = 'Pattern classification is ambiguous';
    } else if (urlInfo.prediction === 'unavailable') {
        iconClass = 'fas fa-times-circle';
        iconStyle = 'mixed';
        title = 'Analysis Unavailable';
        subtitle = 'Cannot classify this URL';
    } else if (urlInfo.prediction === 'error') {
        iconClass = 'fas fa-exclamation-triangle';
        iconStyle = 'suspicious';
        title = 'Analysis Error';
        subtitle = 'An error occurred during analysis';
    }
    
    resultIcon.className = `result-icon ${iconStyle}`;
    resultIcon.innerHTML = `<i class="${iconClass}"></i>`;
    resultTitle.textContent = title;
    resultSubtitle.textContent = subtitle;
    
    // Update result details with tooltips
    const resultDetails = document.getElementById('resultDetails');
    resultDetails.innerHTML = `
        <div class="detail-item">
            <div class="detail-label">
                ${createTooltip('Cluster ID', 'The group number this URL belongs to based on similar structural patterns')}
            </div>
            <div class="detail-value">${urlInfo.cluster_id || 'N/A'}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">
                ${createTooltip('Pattern Similarity', 'How closely this URL matches its cluster pattern (higher = more similar)')}
            </div>
            <div class="detail-value">${(urlInfo.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">
                ${createTooltip('Neighbor Consistency', 'How consistent the pattern is among nearby URLs (higher = more reliable)')}
            </div>
            <div class="detail-value">${(urlInfo.neighbor_confidence * 100).toFixed(1)}%</div>
        </div>
                <div class="detail-item">
                    <div class="detail-label">
                        ${createTooltip('Distance to Center', 'How far this URL is from its cluster center. Lower values mean the URL is more typical/representative of its pattern group, while higher values indicate it\'s more unusual or an outlier.')}
                    </div>
                    <div class="detail-value">${urlInfo.distance_to_centroid ? urlInfo.distance_to_centroid.toFixed(3) : 'N/A'}</div>
                </div>
        <div class="detail-item">
            <div class="detail-label">URL</div>
            <div class="detail-value" style="word-break: break-all; font-size: 14px;">${urlInfo.url}</div>
        </div>
        ${urlInfo.message ? `
        <div class="detail-item" style="grid-column: 1 / -1;">
            <div class="detail-label">Message</div>
            <div class="detail-value" style="font-style: italic; color: rgba(255, 255, 255, 0.7);">${urlInfo.message}</div>
        </div>
        ` : ''}
    `;
    
    // Update neighbors section with improved formatting
    if (urlInfo.nearest_neighbors && urlInfo.nearest_neighbors.length > 0) {
        const neighborsSection = document.getElementById('neighborsSection');
        const neighborsGrid = document.getElementById('neighborsGrid');
        
        neighborsGrid.innerHTML = urlInfo.nearest_neighbors.map(nn => {
            const patternType = nn.label === 'phishing' ? 'Suspicious' : 
                              nn.label === 'legitimate' ? 'Normal' : 'Unknown';
            const badgeClass = nn.label === 'phishing' ? 'suspicious' : 
                             nn.label === 'legitimate' ? 'safe' : 'unknown';
            
            const domain = extractDomain(nn.url);
            const faviconLetter = getFaviconLetter(domain);
            const formattedURL = formatURLDisplay(nn.url);
            // Distance is now normalized (0-1), so similarity = 1 - distance
            const similarity = (1 - nn.distance).toFixed(3);
            const similarityPercent = ((1 - nn.distance) * 100).toFixed(1);
            
            return `
                <div class="neighbor-item" role="article" aria-label="Similar URL: ${domain}">
                    <div class="neighbor-header">
                        <div class="neighbor-favicon" aria-hidden="true">${faviconLetter}</div>
                        <div class="neighbor-domain">${domain}</div>
                        <div class="neighbor-badge ${badgeClass}">${patternType}</div>
                    </div>
                    <div class="neighbor-url">${formattedURL}</div>
                    <div class="neighbor-meta">
                        <i class="fas fa-chart-line" aria-hidden="true"></i> Similarity: ${similarityPercent}% 
                        <span style="margin: 0 8px;">|</span>
                        <i class="fas fa-layer-group" aria-hidden="true"></i> Cluster: ${nn.cluster}
                    </div>
                </div>
            `;
        }).join('');
        
        neighborsSection.style.display = 'block';
    }
    
    // Update charts section with fullscreen capability
    if (data.main_dendrogram || data.url_analysis) {
        const chartsSection = document.getElementById('chartsSection');
        
        if (data.main_dendrogram) {
            document.getElementById('mainDendrogramPlot').innerHTML = `
                <img src="${data.main_dendrogram}" class="chart-image" alt="Hierarchical clustering dendrogram" onclick="openChartFullscreen(this)">
            `;
        }
        if (data.url_analysis) {
            document.getElementById('urlAnalysisPlot').innerHTML = `
                <img src="${data.url_analysis}" class="chart-image" alt="URL cluster pattern analysis" onclick="openChartFullscreen(this)">
            `;
        }
        
        chartsSection.style.display = 'grid';
    }
}

// Chart fullscreen functionality
function openChartFullscreen(imgElement) {
    const modal = document.getElementById('chartModal');
    const modalImg = document.getElementById('chartModalImage');
    modalImg.src = imgElement.src;
    modalImg.alt = imgElement.alt;
    modal.classList.add('active');
    document.body.style.overflow = 'hidden'; // Prevent background scroll
    
    // Focus trap for accessibility
    document.getElementById('chartModalClose').focus();
}

function closeChartFullscreen() {
    const modal = document.getElementById('chartModal');
    modal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scroll
}

// Close modal listeners
document.getElementById('chartModalClose').addEventListener('click', closeChartFullscreen);
document.getElementById('chartModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeChartFullscreen();
    }
});

// Escape key to close modal
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const modal = document.getElementById('chartModal');
        if (modal.classList.contains('active')) {
            closeChartFullscreen();
        }
    }
});

// Utility Functions
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (show) {
        spinner.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    } else {
        spinner.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze';
    }
}

function showResultsSection(show) {
    const resultsSection = document.getElementById('resultsSection');
    if (show) {
        resultsSection.style.display = 'block';
        // Add fade-in animation
        resultsSection.style.opacity = '0';
        resultsSection.style.transform = 'translateY(20px)';
        setTimeout(() => {
            resultsSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            resultsSection.style.opacity = '1';
            resultsSection.style.transform = 'translateY(0)';
        }, 10);
    } else {
        resultsSection.style.display = 'none';
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (urlInput.value.trim()) {
            document.getElementById('urlForm').dispatchEvent(new Event('submit'));
        }
    }
    // Escape to clear input
    if (e.key === 'Escape' && document.activeElement === urlInput) {
        urlInput.value = '';
        urlInput.classList.remove('valid', 'invalid');
        validationIcon.classList.remove('show', 'valid', 'invalid');
        showToast('Input cleared', 'info', 'Cleared');
    }
});


// Add educational tooltips for clustering concepts
function addEducationalTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            showTooltip(this, tooltipText);
        });
        
        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

// Show educational tooltip
function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'educational-tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
}

// Hide educational tooltip
function hideTooltip() {
    const tooltip = document.querySelector('.educational-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Initialize educational features when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    addEducationalTooltips();
});
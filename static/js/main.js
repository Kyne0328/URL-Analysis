let currentUrl = '';
let currentResults = null;

// URL Analysis
document.getElementById('urlForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const url = document.getElementById('urlInput').value.trim();
    if (!url) {
        showToast('Please enter a URL to analyze', 'warning', 'Input Required');
        return;
    }

    // Import validateURL from validation.js
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

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    const urlInput = document.getElementById('urlInput');
    const validationIcon = document.getElementById('validationIcon');

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
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

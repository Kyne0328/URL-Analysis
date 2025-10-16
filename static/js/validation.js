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
        // Import showToast from ui.js will be handled in main.js
    });
});

// URL Validation
function validateURL(url) {
    try {
        // NEW, more comprehensive regex that handles query strings, fragments, and special characters like '%'.
        // This pattern is much more compliant with RFC 3986 for URLs.
        const urlPattern = new RegExp(
            '^(https?:\\/\\/)?' + // protocol
            '((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.)+[a-z]{2,}|' + // domain name
            '((\\d{1,3}\\.){3}\\d{1,3}))' + // OR ip (v4) address
            '(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*' + // port and path
            '(\\?[;&a-z\\d%_.~+=-]*)?' + // query string
            '(\\#[-a-z\\d_]*)?$', // fragment locator
            'i'
        );
        return !!urlPattern.test(url);
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

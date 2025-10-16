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

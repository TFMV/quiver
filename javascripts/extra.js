// Quiver Documentation Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Create mobile bottom navigation
  createMobileBottomNav();
  
  // Enhance code blocks
  enhanceCodeBlocks();
  
  // Add smooth scrolling
  enableSmoothScrolling();
  
  // Add active class to current section in TOC
  highlightTableOfContents();
  
  // Add copy button to code blocks
  addCopyButtons();
  
  // Add version selector
  setupVersionSelector();
  
  // Add search highlighting
  enhanceSearch();
  
  // Add scroll to top button
  addScrollToTopButton();
});

/**
 * Creates a mobile bottom navigation bar for quick access to main sections
 */
function createMobileBottomNav() {
  // Only create on mobile
  if (window.innerWidth > 1220) return;
  
  const mainNav = document.querySelector('.md-tabs');
  if (!mainNav) return;
  
  // Create bottom nav container
  const bottomNav = document.createElement('nav');
  bottomNav.className = 'md-bottom-nav';
  
  // Get main navigation items (limited to 5 for mobile)
  const navItems = Array.from(document.querySelectorAll('.md-tabs__item')).slice(0, 5);
  
  // Create bottom nav items
  navItems.forEach(item => {
    const link = item.querySelector('a');
    if (!link) return;
    
    const navLink = document.createElement('a');
    navLink.className = 'md-bottom-nav__link';
    navLink.href = link.href;
    
    // Check if this is the active link
    if (item.classList.contains('md-tabs__item--active')) {
      navLink.classList.add('md-bottom-nav__link--active');
    }
    
    // Create icon based on link text
    const icon = document.createElement('span');
    icon.className = 'md-bottom-nav__icon';
    
    // Set icon based on link text
    const linkText = link.textContent.trim().toLowerCase();
    if (linkText.includes('home')) {
      icon.innerHTML = '🏠';
    } else if (linkText.includes('guide')) {
      icon.innerHTML = '📚';
    } else if (linkText.includes('api')) {
      icon.innerHTML = '🔌';
    } else if (linkText.includes('example')) {
      icon.innerHTML = '🧩';
    } else if (linkText.includes('advanced')) {
      icon.innerHTML = '🔧';
    } else if (linkText.includes('community')) {
      icon.innerHTML = '👥';
    } else {
      icon.innerHTML = '📄';
    }
    
    // Create label
    const label = document.createElement('span');
    label.className = 'md-bottom-nav__label';
    label.textContent = link.textContent.trim();
    
    navLink.appendChild(icon);
    navLink.appendChild(label);
    bottomNav.appendChild(navLink);
  });
  
  // Add to document
  document.body.appendChild(bottomNav);
}

/**
 * Enhances code blocks with syntax highlighting and line numbers
 */
function enhanceCodeBlocks() {
  const codeBlocks = document.querySelectorAll('pre code');
  
  codeBlocks.forEach(block => {
    // Add line numbers if not already present
    if (!block.classList.contains('line-numbers')) {
      const lines = block.textContent.split('\n').length - 1;
      const lineNumbers = document.createElement('span');
      lineNumbers.className = 'line-numbers-rows';
      
      for (let i = 0; i < lines; i++) {
        const lineSpan = document.createElement('span');
        lineNumbers.appendChild(lineSpan);
      }
      
      block.parentNode.classList.add('line-numbers');
      block.parentNode.appendChild(lineNumbers);
    }
  });
}

/**
 * Enables smooth scrolling for anchor links
 */
function enableSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        
        window.scrollTo({
          top: targetElement.offsetTop - 100,
          behavior: 'smooth'
        });
        
        // Update URL without scrolling
        history.pushState(null, null, targetId);
      }
    });
  });
}

/**
 * Highlights the current section in the table of contents
 */
function highlightTableOfContents() {
  const tocLinks = document.querySelectorAll('.md-nav__link');
  const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
  
  if (tocLinks.length === 0 || headings.length === 0) return;
  
  // Create a map of heading IDs to their positions
  const headingPositions = Array.from(headings).map(heading => {
    return {
      id: heading.id,
      top: heading.offsetTop
    };
  });
  
  // Update active TOC item on scroll
  window.addEventListener('scroll', function() {
    const scrollPosition = window.scrollY + 150;
    
    // Find the current heading
    let currentHeadingId = '';
    for (let i = 0; i < headingPositions.length; i++) {
      if (headingPositions[i].top <= scrollPosition) {
        currentHeadingId = headingPositions[i].id;
      } else {
        break;
      }
    }
    
    // Update active TOC item
    if (currentHeadingId) {
      tocLinks.forEach(link => {
        link.classList.remove('md-nav__link--active-scroll');
        
        const href = link.getAttribute('href');
        if (href && href.endsWith('#' + currentHeadingId)) {
          link.classList.add('md-nav__link--active-scroll');
        }
      });
    }
  });
}

/**
 * Adds copy buttons to code blocks
 */
function addCopyButtons() {
  const codeBlocks = document.querySelectorAll('pre code');
  
  codeBlocks.forEach(block => {
    // Check if copy button already exists
    if (block.parentNode.querySelector('.copy-button')) return;
    
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    
    button.addEventListener('click', function() {
      const code = block.textContent;
      navigator.clipboard.writeText(code).then(() => {
        // Show success feedback
        button.textContent = 'Copied!';
        button.classList.add('copied');
        
        // Reset after 2 seconds
        setTimeout(() => {
          button.textContent = 'Copy';
          button.classList.remove('copied');
        }, 2000);
      });
    });
    
    // Add button to pre element
    block.parentNode.appendChild(button);
  });
}

// Add styles for the copy button
const style = document.createElement('style');
style.textContent = `
  .copy-button {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.8rem;
    color: var(--quiver-text-light);
    background-color: var(--quiver-background);
    border: 1px solid var(--quiver-border);
    border-radius: 4px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  
  pre:hover .copy-button {
    opacity: 1;
  }
  
  .copy-button.copied {
    color: white;
    background-color: var(--quiver-primary);
    border-color: var(--quiver-primary);
  }
  
  .md-nav__link--active-scroll {
    color: var(--quiver-primary) !important;
    font-weight: bold;
  }
  
  pre {
    position: relative;
  }
`;

document.head.appendChild(style);

// Setup version selector with animation
function setupVersionSelector() {
  const versionSelector = document.querySelector('.md-version');
  if (!versionSelector) return;
  
  versionSelector.addEventListener('change', function() {
    // Add a loading animation
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'version-loading';
    loadingIndicator.innerHTML = '<span>Loading...</span>';
    loadingIndicator.style.position = 'fixed';
    loadingIndicator.style.top = '50%';
    loadingIndicator.style.left = '50%';
    loadingIndicator.style.transform = 'translate(-50%, -50%)';
    loadingIndicator.style.padding = '1rem 2rem';
    loadingIndicator.style.backgroundColor = '#673ab7';
    loadingIndicator.style.color = 'white';
    loadingIndicator.style.borderRadius = '0.5rem';
    loadingIndicator.style.zIndex = '9999';
    
    document.body.appendChild(loadingIndicator);
  });
}

// Enhance search with better highlighting
function enhanceSearch() {
  const searchInput = document.querySelector('.md-search__input');
  if (!searchInput) return;
  
  searchInput.addEventListener('focus', function() {
    // Add a subtle animation
    this.style.transition = 'all 0.3s ease';
    this.style.boxShadow = '0 0 0 3px rgba(103, 58, 183, 0.2)';
  });
  
  searchInput.addEventListener('blur', function() {
    this.style.boxShadow = 'none';
  });
}

// Add a scroll to top button
function addScrollToTopButton() {
  const button = document.createElement('button');
  button.className = 'scroll-to-top';
  button.innerHTML = '&uarr;';
  button.setAttribute('aria-label', 'Scroll to top');
  button.setAttribute('title', 'Scroll to top');
  
  // Style the button
  button.style.position = 'fixed';
  button.style.bottom = '2rem';
  button.style.right = '2rem';
  button.style.width = '3rem';
  button.style.height = '3rem';
  button.style.borderRadius = '50%';
  button.style.backgroundColor = '#673ab7';
  button.style.color = 'white';
  button.style.border = 'none';
  button.style.fontSize = '1.5rem';
  button.style.cursor = 'pointer';
  button.style.opacity = '0';
  button.style.transition = 'opacity 0.3s ease';
  button.style.zIndex = '100';
  button.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.2)';
  
  document.body.appendChild(button);
  
  // Show/hide the button based on scroll position
  window.addEventListener('scroll', function() {
    if (window.scrollY > 300) {
      button.style.opacity = '1';
    } else {
      button.style.opacity = '0';
    }
  });
  
  // Scroll to top when clicked
  button.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
  
  // Add hover effect
  button.addEventListener('mouseenter', function() {
    this.style.backgroundColor = '#ffc107';
    this.style.transform = 'scale(1.1)';
  });
  
  button.addEventListener('mouseleave', function() {
    this.style.backgroundColor = '#673ab7';
    this.style.transform = 'scale(1)';
  });
}

// Add a simple easter egg
const konami = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];
let konamiIndex = 0;

document.addEventListener('keydown', function(e) {
  if (e.keyCode === konami[konamiIndex++]) {
    if (konamiIndex === konami.length) {
      // Easter egg activated!
      document.body.style.transition = 'all 0.5s ease';
      document.body.style.backgroundColor = '#673ab7';
      
      setTimeout(() => {
        document.body.style.backgroundColor = '';
        
        // Show a fun message
        const message = document.createElement('div');
        message.textContent = 'You found the Quiver easter egg! 🏹';
        message.style.position = 'fixed';
        message.style.top = '50%';
        message.style.left = '50%';
        message.style.transform = 'translate(-50%, -50%)';
        message.style.padding = '1rem 2rem';
        message.style.backgroundColor = '#673ab7';
        message.style.color = 'white';
        message.style.borderRadius = '0.5rem';
        message.style.zIndex = '9999';
        message.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.3)';
        
        document.body.appendChild(message);
        
        setTimeout(() => {
          message.style.opacity = '0';
          setTimeout(() => {
            document.body.removeChild(message);
          }, 500);
        }, 3000);
      }, 500);
      
      konamiIndex = 0;
    }
  } else {
    konamiIndex = 0;
  }
}); 
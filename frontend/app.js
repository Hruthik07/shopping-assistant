// Configuration - Auto-detect API URL based on current host
const API_BASE_URL = (() => {
    // In production, use the same origin (relative URLs)
    // In development, use localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:3565';
    }
    // Production: use same origin (no hardcoded URL)
    return window.location.origin;
})();
let sessionId = null;
let currentPersona = 'friendly';
let currentTone = 'warm';
let conversationFlow = {
    currentStep: 'greeting',
    steps: ['greeting', 'understanding', 'searching', 'recommending', 'closing']
};

// DOM Elements (will be initialized after DOM loads)
let chatMessages, chatInput, sendButton, productsList, productsPanel;
let statusBar, personaSelect, toneSelect, applySettingsBtn, productCount;
let welcomeSection, settingsToggle, sidebar, closeSidebar, mainContainer, quickActionBtns;
let chatResizeToggle, resizeIcon;
let chatResizer, isResizing = false;
let voiceButton, voiceOutputToggle, voiceRateSlider, voicePitchSlider, voiceVolumeSlider;
let voiceRateValue, voicePitchValue, voiceVolumeValue;
let voiceAssistant = null; // Voice assistant instance

// Chat size state: 'normal', 'large', 'compact'
let chatSize = 'normal';

// Debug helper function (disabled by default)
// Enable with: localStorage.setItem('DEBUG_ENABLED', 'true')
// Optional remote sink: localStorage.setItem('DEBUG_INGEST_URL', 'https://.../ingest')
function debugLog(location, message, data, hypothesisId) {
    if (localStorage.getItem('DEBUG_ENABLED') !== 'true') return;
    try {
        const payload = {
            location,
            message,
            data,
            timestamp: Date.now(),
            sessionId: sessionId || 'no-session',
            hypothesisId
        };
        const ingestUrl = localStorage.getItem('DEBUG_INGEST_URL');
        if (ingestUrl && /^https?:\/\//i.test(ingestUrl)) {
            fetch(ingestUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(() => {});
        } else {
            // Default: log locally only (no network calls in production)
            console.debug('[DEBUG]', payload);
        }
    } catch (e) {
        // Silently ignore debug errors
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('[INIT] DOM Content Loaded - Starting initialization...');
    
    // Get DOM elements
    chatMessages = document.getElementById('chat-messages');
    chatInput = document.getElementById('chat-input');
    sendButton = document.getElementById('send-button');
    productsList = document.getElementById('products-list');
    productsPanel = document.getElementById('products-panel');
    statusBar = document.getElementById('status');
    personaSelect = document.getElementById('persona-select');
    toneSelect = document.getElementById('tone-select');
    applySettingsBtn = document.getElementById('apply-settings');
    productCount = document.getElementById('product-count');
    welcomeSection = document.getElementById('welcome-section');
    settingsToggle = document.getElementById('settings-toggle');
    sidebar = document.getElementById('sidebar');
    closeSidebar = document.getElementById('close-sidebar');
    mainContainer = document.querySelector('.main-container');
    quickActionBtns = document.querySelectorAll('.quick-action-btn');
    chatResizeToggle = document.getElementById('chat-resize-toggle');
    resizeIcon = document.getElementById('resize-icon');
    chatResizer = document.getElementById('chat-resizer');
    
    // Voice assistant elements
    voiceButton = document.getElementById('voice-button');
    voiceOutputToggle = document.getElementById('voice-output-toggle');
    
    // Initialize interactive background effects
    initInteractiveEffects();
    voiceRateSlider = document.getElementById('voice-rate');
    voicePitchSlider = document.getElementById('voice-pitch');
    voiceVolumeSlider = document.getElementById('voice-volume');
    voiceRateValue = document.getElementById('voice-rate-value');
    voicePitchValue = document.getElementById('voice-pitch-value');
    voiceVolumeValue = document.getElementById('voice-volume-value');
    
    // Check if all required elements exist
    const missingElements = [];
    if (!chatMessages) missingElements.push('chat-messages');
    if (!chatInput) missingElements.push('chat-input');
    if (!sendButton) missingElements.push('send-button');
    if (!productsList) missingElements.push('products-list');
    if (!productsPanel) missingElements.push('products-panel');
    if (!statusBar) missingElements.push('status');
    
    if (missingElements.length > 0) {
        console.error('[INIT] Critical DOM elements missing:', missingElements);
        alert('Error: Missing required elements: ' + missingElements.join(', ') + '\nPlease check the HTML structure.');
        return;
    }
    
    console.log('[INIT] All critical DOM elements found');
    
    // Initialize voice assistant (class defined later in file)
    try {
        voiceAssistant = new VoiceAssistant();
    } catch (e) {
        console.error('Failed to initialize VoiceAssistant:', e);
        voiceAssistant = null;
    }
    
    // Check browser compatibility
    const voiceSupport = checkVoiceSupport();
    if (!voiceSupport.fullySupported) {
        console.warn('Voice features partially supported:', voiceSupport);
        if (!voiceSupport.speechToText && voiceButton) {
            voiceButton.disabled = true;
            voiceButton.title = 'Voice input not supported in this browser';
        }
    }
    
    // Load voices (some browsers need this)
    if (window.speechSynthesis) {
        // Load voices immediately if available
        if (window.speechSynthesis.getVoices().length > 0) {
            // Voices already loaded
        } else {
            // Wait for voices to load
            window.speechSynthesis.onvoiceschanged = () => {
                // Voices loaded
            };
        }
    }
    
    // Generate or get session ID
    sessionId = localStorage.getItem('sessionId') || generateSessionId();
    localStorage.setItem('sessionId', sessionId);
    
    // Load saved settings
    loadSettings();
    
    // Load saved chat size
    chatSize = localStorage.getItem('chatSize') || 'normal';
    applyChatSize(chatSize);
    
    // Event listeners
    if (sendButton) {
        console.log('[INIT] Attaching send button click listener');
        sendButton.addEventListener('click', () => {
            console.log('[CLICK] Send button clicked');
            sendMessage();
        });
    } else {
        console.error('[INIT] Send button not found!');
    }
    
    if (chatInput) {
        console.log('[INIT] Attaching chat input keypress listener');
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('[KEYPRESS] Enter pressed, sending message');
                sendMessage();
            }
        });
        // Focus input
        chatInput.focus();
        console.log('[INIT] Chat input focused');
    } else {
        console.error('[INIT] Chat input not found!');
    }
    
    if (applySettingsBtn) {
        applySettingsBtn.addEventListener('click', applySettings);
    }
    
    // Chat resize toggle (button)
    if (chatResizeToggle) {
        chatResizeToggle.addEventListener('click', toggleChatSize);
    }
    
    // Draggable resizer
    if (chatResizer) {
        initResizer();
    }
    
    // Sidebar toggle with active state
    if (settingsToggle && mainContainer) {
        settingsToggle.addEventListener('click', () => {
            const isOpen = mainContainer.classList.contains('sidebar-open');
            if (isOpen) {
                mainContainer.classList.remove('sidebar-open');
                settingsToggle.classList.remove('active');
                settingsToggle.setAttribute('aria-expanded', 'false');
            } else {
                mainContainer.classList.add('sidebar-open');
                settingsToggle.classList.add('active');
                settingsToggle.setAttribute('aria-expanded', 'true');
            }
        });
    }
    
    if (closeSidebar && mainContainer) {
        closeSidebar.addEventListener('click', () => {
            mainContainer.classList.remove('sidebar-open');
            if (settingsToggle) {
                settingsToggle.classList.remove('active');
                settingsToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }
    
    // Header scroll detection
    const topNav = document.querySelector('.top-nav');
    if (topNav) {
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
            if (currentScroll > 10) {
                topNav.classList.add('scrolled');
            } else {
                topNav.classList.remove('scrolled');
            }
            lastScroll = currentScroll;
        });
    }
    
    // Quick action buttons
    if (quickActionBtns && quickActionBtns.length > 0) {
        quickActionBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.getAttribute('data-query');
                if (chatInput && query) {
                    chatInput.value = query;
                    sendMessage();
                }
            });
        });
    }
    
    // Voice button event listener
    if (voiceButton && voiceAssistant) {
        voiceButton.addEventListener('click', () => {
            voiceAssistant.toggleVoiceInput();
        });
    }
    
    // Voice settings event listeners
    if (voiceOutputToggle) {
        // Load saved voice output setting
        const voiceOutputEnabled = localStorage.getItem('voiceOutputEnabled') === 'true';
        voiceOutputToggle.checked = voiceOutputEnabled;
        
        // Show/hide voice settings based on toggle
        const voiceSettingsGroups = document.querySelectorAll('.voice-settings-group');
        voiceSettingsGroups.forEach(group => {
            group.style.display = voiceOutputEnabled ? 'flex' : 'none';
        });
        
        voiceOutputToggle.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            localStorage.setItem('voiceOutputEnabled', enabled.toString());
            voiceSettingsGroups.forEach(group => {
                group.style.display = enabled ? 'flex' : 'none';
            });
        });
    }
    
    // Voice rate slider
    if (voiceRateSlider && voiceRateValue) {
        const savedRate = localStorage.getItem('voiceRate') || '1.0';
        voiceRateSlider.value = savedRate;
        voiceRateValue.textContent = `${savedRate}x`;
        
        voiceRateSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            voiceRateValue.textContent = `${value}x`;
            localStorage.setItem('voiceRate', value);
        });
    }
    
    // Voice pitch slider
    if (voicePitchSlider && voicePitchValue) {
        const savedPitch = localStorage.getItem('voicePitch') || '1.0';
        voicePitchSlider.value = savedPitch;
        voicePitchValue.textContent = savedPitch;
        
        voicePitchSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            voicePitchValue.textContent = value;
            localStorage.setItem('voicePitch', value);
        });
    }
    
    // Voice volume slider
    if (voiceVolumeSlider && voiceVolumeValue) {
        const savedVolume = localStorage.getItem('voiceVolume') || '0.8';
        voiceVolumeSlider.value = savedVolume;
        voiceVolumeValue.textContent = `${Math.round(savedVolume * 100)}%`;
        
        voiceVolumeSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            voiceVolumeValue.textContent = `${Math.round(value * 100)}%`;
            localStorage.setItem('voiceVolume', value);
        });
    }
    
    // Check API connection after DOM loads
    setTimeout(checkConnection, 100);
});

function generateSessionId() {
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

function loadSettings() {
    const savedPersona = localStorage.getItem('persona') || 'friendly';
    const savedTone = localStorage.getItem('tone') || 'warm';
    
    if (personaSelect) personaSelect.value = savedPersona;
    if (toneSelect) toneSelect.value = savedTone;
    
    currentPersona = savedPersona;
    currentTone = savedTone;
    
    // Initialize voice settings defaults if not set
    if (!localStorage.getItem('voiceRate')) localStorage.setItem('voiceRate', '1.0');
    if (!localStorage.getItem('voicePitch')) localStorage.setItem('voicePitch', '1.0');
    if (!localStorage.getItem('voiceVolume')) localStorage.setItem('voiceVolume', '0.8');
    if (!localStorage.getItem('voiceOutputEnabled')) localStorage.setItem('voiceOutputEnabled', 'false');
}

// Chat resize functionality
function toggleChatSize() {
    // Cycle through sizes: normal -> large -> compact -> normal
    if (chatSize === 'normal') {
        chatSize = 'large';
    } else if (chatSize === 'large') {
        chatSize = 'compact';
    } else {
        chatSize = 'normal';
    }
    
    applyChatSize(chatSize);
    localStorage.setItem('chatSize', chatSize);
}

function applyChatSize(size) {
    if (!mainContainer) return;
    
    // Remove all size classes
    mainContainer.classList.remove('chat-normal', 'chat-large', 'chat-compact');
    
    // Add the new size class
    mainContainer.classList.add(`chat-${size}`);
    
    // Update icon to indicate current size
    if (resizeIcon) {
        if (size === 'large') {
            resizeIcon.innerHTML = '<path d="M7 14l5-5 5 5H7z" fill="currentColor"/>'; // Up arrow
        } else if (size === 'compact') {
            resizeIcon.innerHTML = '<path d="M7 10l5 5 5-5H7z" fill="currentColor"/>'; // Down arrow
        } else {
            resizeIcon.innerHTML = '<path d="M7 10l5 5 5-5H7z" fill="currentColor"/>'; // Default
        }
    }
}

// Draggable resizer functionality
function initResizer() {
    if (!chatResizer || !mainContainer) return;
    
    const chatArea = document.querySelector('.chat-area');
    const productsPanel = document.getElementById('products-panel');
    
    if (!chatArea || !productsPanel) return;
    
    let startX = 0;
    let startChatWidth = 0;
    
    // Load saved width
    const savedChatWidth = localStorage.getItem('chatWidth');
    if (savedChatWidth) {
        const width = parseFloat(savedChatWidth);
        chatArea.style.width = `${width}%`;
        productsPanel.style.width = `${100 - width}%`;
    } else {
        // Set default widths - more compact, use percentages
        chatArea.style.width = '55%';
        productsPanel.style.width = '45%';
        productsPanel.style.flexGrow = '1'; // Allow products to take remaining space
    }
    
    chatResizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startChatWidth = chatArea.offsetWidth;
        
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        document.body.style.pointerEvents = 'none';
        chatResizer.style.pointerEvents = 'auto';
        e.preventDefault();
        e.stopPropagation();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const containerWidth = mainContainer.offsetWidth;
        const diff = e.clientX - startX;
        const newChatWidth = startChatWidth + diff;
        
        // Calculate percentages
        const chatPercent = (newChatWidth / containerWidth) * 100;
        const productsPercent = 100 - chatPercent;
        
        // Apply constraints (in pixels for better control)
        const minChatWidth = 300; // Minimum 300px for chat
        const maxChatWidth = containerWidth - 250; // Maximum (leave 250px for products)
        const newChatWidthPx = newChatWidth;
        
        if (newChatWidthPx >= minChatWidth && newChatWidthPx <= maxChatWidth) {
            chatArea.style.width = `${newChatWidthPx}px`;
            productsPanel.style.width = `${containerWidth - newChatWidthPx - 6}px`; // Subtract resizer width
            productsPanel.style.flexGrow = '0'; // Prevent flex-grow when manually sized
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.body.style.pointerEvents = '';
            
            // Save the width preference as percentage
            if (chatArea && mainContainer) {
                const chatPercent = (chatArea.offsetWidth / mainContainer.offsetWidth) * 100;
                localStorage.setItem('chatWidth', chatPercent.toString());
            }
        }
    });
}

function applySettings() {
    if (!personaSelect || !toneSelect || !applySettingsBtn) {
        console.error('Settings elements not found');
        return;
    }
    
    currentPersona = personaSelect.value;
    currentTone = toneSelect.value;
    
    localStorage.setItem('persona', currentPersona);
    localStorage.setItem('tone', currentTone);
    
    // Visual feedback
    applySettingsBtn.textContent = 'Applied!';
    applySettingsBtn.style.background = '#10b981';
    setTimeout(() => {
        applySettingsBtn.textContent = 'Apply Settings';
        applySettingsBtn.style.background = '';
    }, 2000);
}

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

async function sendMessage() {
    console.log('[SEND] sendMessage called');
    
    const message = chatInput.value.trim();
    if (!message) {
        console.log('[SEND] Empty message, returning');
        return;
    }
    
    console.log('[SEND] Message:', message.substring(0, 50) + '...');
    
    // Validate message length
    if (message.length > 1000) {
        addMessage('Message is too long. Please keep it under 1000 characters.', 'bot');
        return;
    }
    
    // Disable input during processing
    if (chatInput) chatInput.disabled = true;
    if (sendButton) {
        sendButton.disabled = true;
        sendButton.classList.add('loading');
    }
    
    // Hide welcome section after first message
    if (welcomeSection) {
        welcomeSection.style.display = 'none';
    }
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    chatInput.value = '';
    
    // Show loading
    const loadingId = addMessage('', 'bot', true);
    updateStatus('Thinking...', 'processing');
    showTypingIndicator();
    
    console.log('[SEND] Making API request to:', `${API_BASE_URL}/api/chat/`);
    console.log('[SEND] Session ID:', sessionId);
    
    let lastError = null;
    
    // Retry logic
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
            
            const response = await fetch(`${API_BASE_URL}/api/chat/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId,
                    persona: currentPersona,
                    tone: currentTone
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                // Don't retry on 4xx errors (client errors)
                if (response.status >= 400 && response.status < 500) {
                    const errorText = await response.text().catch(() => 'Unknown error');
                    console.error('[SEND] HTTP error response:', errorText);
                    throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
                }
                // Retry on 5xx errors (server errors)
                const errorText = await response.text().catch(() => 'Unknown error');
                console.error('[SEND] HTTP error response:', errorText);
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }
            
            console.log('[SEND] Response OK, parsing JSON...');
            let data;
            try {
                const responseText = await response.text();
                console.log('[SEND] Response text length:', responseText.length);
                console.log('[SEND] Response text preview:', responseText.substring(0, 200));
                data = JSON.parse(responseText);
                console.log('[SEND] JSON parsed successfully');
            } catch (parseError) {
                console.error('[SEND] JSON parse error:', parseError);
                console.error('[SEND] Response was:', await response.text().catch(() => 'Could not read response'));
                throw new Error('Failed to parse response as JSON: ' + parseError.message);
            }
            
            console.log('[SEND] API response received:', {
                hasResponse: !!data.response,
                responseLength: data.response?.length || 0,
                hasProducts: !!data.products,
                productsCount: data.products?.length || 0,
                sessionId: data.session_id,
                keys: Object.keys(data)
            });
            
            debugLog('app.js:API response received', 'API response received', {
                hasProducts: !!data.products,
                productsCount: data.products?.length || 0,
                products: data.products?.map(p => ({
                    name: p.name || p.title,
                    url: p.product_url || p.link || p.url || p.productUrl
                })) || [],
                responsePreview: data.response?.substring(0, 200)
            }, 'A');
            
            // Validate response data
            if (!data || typeof data !== 'object') {
                console.error('[SEND] Invalid response format:', data);
                throw new Error('Invalid response format');
            }
            
            if (!data.response) {
                console.error('[SEND] Response missing "response" field:', data);
                throw new Error('Response missing required "response" field');
            }
            
            console.log('[SEND] Response validation passed');
            
            // Update session ID from response if it changed
            if (data.session_id && data.session_id !== sessionId) {
                sessionId = data.session_id;
                localStorage.setItem('sessionId', sessionId);
            }
            
            // Remove loading message
            removeMessage(loadingId);
            
            // Inject product URLs into response text before displaying
            let responseText = data.response || 'Sorry, I couldn\'t process that request.';
            if (!responseText || typeof responseText !== 'string') {
                responseText = 'Sorry, I couldn\'t process that request.';
            }
            
            debugLog('app.js:470', 'Before injectProductLinks', {
                responseTextPreview: responseText.substring(0, 300),
                hasPlaceholders: responseText.includes('[product_url]') || responseText.includes('[Product URL]')
            }, 'E');
            
            if (data.products && Array.isArray(data.products) && data.products.length > 0) {
                responseText = injectProductLinks(responseText, data.products);
                
                debugLog('app.js:473', 'After injectProductLinks', {
                    responseTextPreview: responseText.substring(0, 300),
                    hasMarkdownLinks: responseText.includes('[View Product](')
                }, 'D');
            }
            
            // Add bot response
            console.log('[SEND] Adding bot message, length:', responseText.length);
            console.log('[SEND] Response text preview:', responseText.substring(0, 100));
            
            const messageResult = addMessage(responseText, 'bot');
            console.log('[SEND] Message added result:', messageResult);
            
            // Check if message was actually added to DOM (messageResult can be null for non-loading messages)
            // Verify by checking if chatMessages has children
            if (!chatMessages || chatMessages.children.length === 0) {
                console.error('[SEND] Failed to add message to DOM - chatMessages is empty!');
                throw new Error('Failed to add message to DOM');
            }
            
            // Message was successfully added (even if messageResult is null, that's OK for non-loading messages)
            console.log('[SEND] Message successfully added to DOM');
            
            // Display products
            if (data.products && Array.isArray(data.products) && data.products.length > 0) {
                console.log('[SEND] Displaying', data.products.length, 'products');
                displayProducts(data.products);
                showToast(`Found ${data.products.length} product${data.products.length > 1 ? 's' : ''}`, 'success', 3000);
            } else {
                console.log('[SEND] No products to display');
                clearProducts();
            }
            
            // Speak the response if voice output is enabled
            const voiceEnabled = localStorage.getItem('voiceOutputEnabled') === 'true';
            if (voiceAssistant && voiceEnabled && responseText) {
                // Small delay for better UX
                setTimeout(() => {
                    voiceAssistant.speak(responseText, {
                        rate: parseFloat(localStorage.getItem('voiceRate') || '1.0'),
                        pitch: parseFloat(localStorage.getItem('voicePitch') || '1.0'),
                        volume: parseFloat(localStorage.getItem('voiceVolume') || '0.8')
                    });
                }, 500);
            }
            
            updateStatus('Ready', 'ready');
            hideTypingIndicator();
            
            // Re-enable input
            if (chatInput) chatInput.disabled = false;
            if (sendButton) {
                sendButton.disabled = false;
                sendButton.classList.remove('loading');
            }
            
            console.log('[SEND] Successfully processed response, exiting retry loop');
            return; // Success, exit retry loop
            
        } catch (error) {
            console.error('[SEND] Error in attempt', attempt + 1, ':', error);
            console.error('[SEND] Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
            lastError = error;
            
            // Don't retry on AbortError (timeout) or client errors
            if (error.name === 'AbortError' || (error.message && error.message.includes('4'))) {
                break;
            }
            
            // Wait before retry (exponential backoff)
            if (attempt < MAX_RETRIES - 1) {
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (attempt + 1)));
                updateStatus(`Retrying... (${attempt + 2}/${MAX_RETRIES})`);
            }
        }
    }
    
    // All retries failed
    removeMessage(loadingId);
    if (lastError && lastError.name === 'AbortError') {
        const errorMsg = 'Request timed out. Please try again.';
        addMessage(errorMsg, 'bot');
        showToast(errorMsg, 'error');
        updateStatus('Timeout');
    } else {
        console.error('Error:', lastError);
        const errorMsg = 'I\'m sorry, I encountered an error processing your request. Please try again.';
        addMessage(errorMsg, 'bot');
        showToast(errorMsg, 'error');
        updateStatus('Error');
    }
    
    // Re-enable input
    if (chatInput) chatInput.disabled = false;
    if (sendButton) {
        sendButton.disabled = false;
        sendButton.classList.remove('loading');
    }
    hideTypingIndicator();
    updateStatus('Ready', 'ready');
}

function injectProductLinks(responseText, products) {
    debugLog('app.js:536', 'injectProductLinks entry', {
        productsCount: products.length,
        products: products.map(p => ({
            name: p.name || p.title,
            url: p.product_url || p.link || p.url || p.productUrl,
            urlType: typeof (p.product_url || p.link || p.url || p.productUrl)
        })),
        responseTextPreview: responseText.substring(0, 200)
    }, 'A');
    
    // Replace [product_url], [Product URL], or other placeholders with actual clickable markdown links
    // Match each product recommendation to its corresponding product URL
    let modifiedText = responseText;
    
    // Strategy: Match products by finding product names in the text and replacing the nearest placeholder
    // Split text into sections by numbered recommendations (1., 2., 3., etc.)
    const sections = modifiedText.split(/(?=\d+\.\s+)/);
    
    debugLog('app.js:545', 'Sections split', {
        sectionsCount: sections.length,
        sectionsPreview: sections.slice(0, 3).map(s => s.substring(0, 100))
    }, 'D');
    
    if (sections.length > 1) {
        // We have numbered sections - match each section to a product
        const processedSections = sections.map((section, sectionIndex) => {
            // Skip the first section if it doesn't start with a number (intro text)
            if (sectionIndex === 0 && !/^\d+\./.test(section)) {
                return section;
            }
            
            // Find which product this section corresponds to
            // Try to match by product name in the section
            let matchedProduct = null;
            let bestMatchIndex = -1;
            let bestMatchScore = 0;
            
            for (let i = 0; i < products.length; i++) {
                const product = products[i];
                const productName = (product.name || product.title || '').toLowerCase();
                const sectionLower = section.toLowerCase();
                
                // Check if product name appears in this section
                if (productName && sectionLower.includes(productName)) {
                    const matchScore = productName.length;
                    if (matchScore > bestMatchScore) {
                        bestMatchScore = matchScore;
                        bestMatchIndex = i;
                        matchedProduct = product;
                    }
                }
            }
            
            // If no name match, try to match by position (section index - 1)
            if (!matchedProduct && sectionIndex > 0) {
                const productIndex = sectionIndex - 1;
                if (productIndex < products.length) {
                    matchedProduct = products[productIndex];
                }
            }
            
            // Replace placeholders in this section with the matched product's URL
            if (matchedProduct) {
                const productUrl = matchedProduct.product_url || matchedProduct.link || matchedProduct.url || matchedProduct.productUrl;
                
                debugLog('app.js:584', 'Product matched', {
                    sectionIndex: sectionIndex,
                    matchedProductName: matchedProduct.name || matchedProduct.title,
                    originalUrl: productUrl,
                    urlLength: productUrl?.length
                }, 'D');
                
                if (productUrl) {
                    // Clean and validate the URL
                    let cleanUrl = productUrl.trim();
                    
                    debugLog('app.js:590', 'URL before cleaning', {
                        originalUrl: productUrl,
                        trimmedUrl: cleanUrl
                    }, 'B');
                    
                    // Remove any trailing punctuation or whitespace
                    cleanUrl = cleanUrl.replace(/[.,;:!?\s]+$/, '');
                    
                    debugLog('app.js:593', 'URL after cleaning', {
                        cleanUrl: cleanUrl,
                        isValid: !!cleanUrl.match(/^https?:\/\//i)
                    }, 'B');
                    
                    // Validate URL format - must start with http:// or https://
                    if (!cleanUrl.match(/^https?:\/\//i)) {
                        // If it's not a valid URL, skip this product
                        console.warn('Invalid product URL format:', cleanUrl);
                        debugLog('app.js:597', 'Invalid URL rejected', { cleanUrl: cleanUrl }, 'A');
                        return section;
                    }
                    
                    // Use a simple link text instead of product name to avoid markdown issues
                    const linkText = `[View Product](${cleanUrl})`;
                    
                    debugLog('app.js:600', 'Link text created', { linkText: linkText, cleanUrl: cleanUrl }, 'E');
                    
                    // Replace placeholders in this section only (not globally)
                    let sectionText = section;
                    const beforeReplace = sectionText;
                    sectionText = sectionText.replace(/Buy here:\s*\[product_url\]/gi, `Buy here: ${linkText}`);
                    sectionText = sectionText.replace(/Buy here:\s*\[Product URL\]/gi, `Buy here: ${linkText}`);
                    sectionText = sectionText.replace(/Buy here:\s*\[PRODUCT_URL\]/gi, `Buy here: ${linkText}`);
                    sectionText = sectionText.replace(/\[product_url\]/gi, linkText);
                    sectionText = sectionText.replace(/\[Product URL\]/g, linkText);
                    sectionText = sectionText.replace(/\[PRODUCT_URL\]/g, linkText);
                    sectionText = sectionText.replace(/\[Google Shopping Link\]/gi, linkText);
                    sectionText = sectionText.replace(/\[Product Link\]/gi, linkText);
                    
                    debugLog('app.js:612', 'Section after replacement', {
                        beforeReplace: beforeReplace.substring(0, 150),
                        afterReplace: sectionText.substring(0, 150),
                        wasReplaced: beforeReplace !== sectionText
                    }, 'E');
                    
                    return sectionText;
                }
            }
            
            return section;
        });
        
        modifiedText = processedSections.join('');
        
        debugLog('app.js:620', 'injectProductLinks exit (sections)', {
            finalTextPreview: modifiedText.substring(0, 300),
            hasMarkdownLinks: modifiedText.includes('[View Product](')
        }, 'D');
    } else {
        // No numbered sections - use fallback: replace in order (first occurrence gets first product, etc.)
        let placeholderCount = 0;
        modifiedText = modifiedText.replace(/Buy here:\s*\[(?:product_url|Product URL|PRODUCT_URL)\]/gi, (match) => {
            if (placeholderCount < products.length) {
                const product = products[placeholderCount];
                const productUrl = product.product_url || product.link || product.url || product.productUrl;
                if (productUrl) {
                    // Clean and validate the URL
                    let cleanUrl = productUrl.trim();
                    cleanUrl = cleanUrl.replace(/[.,;:!?\s]+$/, '');
                    
                    // Validate URL format
                    if (cleanUrl.match(/^https?:\/\//i)) {
                        const linkText = `[View Product](${cleanUrl})`;
                        placeholderCount++;
                        return `Buy here: ${linkText}`;
                    }
                }
            }
            return match;
        });
        
        // Also handle standalone placeholders
        placeholderCount = 0;
        modifiedText = modifiedText.replace(/\[(?:product_url|Product URL|PRODUCT_URL)\]/gi, (match) => {
            if (placeholderCount < products.length) {
                const product = products[placeholderCount];
                const productUrl = product.product_url || product.link || product.url || product.productUrl;
                if (productUrl) {
                    // Clean and validate the URL
                    let cleanUrl = productUrl.trim();
                    cleanUrl = cleanUrl.replace(/[.,;:!?\s]+$/, '');
                    
                    // Validate URL format
                    if (cleanUrl.match(/^https?:\/\//i)) {
                        const linkText = `[View Product](${cleanUrl})`;
                        placeholderCount++;
                        return linkText;
                    }
                }
            }
            return match;
        });
    }
    
    debugLog('app.js:650', 'injectProductLinks exit', {
        finalTextPreview: modifiedText.substring(0, 300),
        hasMarkdownLinks: modifiedText.includes('[View Product](')
    }, 'A');
    
    return modifiedText;
}

function formatMessage(text) {
    // Sanitize input to prevent XSS
    if (!text || typeof text !== 'string') {
        return '';
    }
    
    // Escape HTML first to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Helper function to escape special regex characters
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    // CRITICAL FIX: Process markdown links BEFORE escaping HTML
    // This prevents double-encoding of URLs (e.g., & becomes &amp; then &amp;amp;)
    // Convert markdown links directly to HTML (with proper escaping of URL and text)
    let processedText = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
        debugLog('app.js:747', 'Markdown link found (pre-escape)', {
            match: match.substring(0, 100),
            linkText: linkText,
            originalUrl: url.substring(0, 100)
        }, 'C');
        
        // Clean up URL (remove any trailing characters that might break the link)
        let cleanUrl = url.trim();
        // Remove trailing punctuation that might have been included
        cleanUrl = cleanUrl.replace(/[.,;:!?\s]+$/, '');
        // Remove any parentheses that might have been included in the URL
        cleanUrl = cleanUrl.replace(/[()]+$/, '');
        
        // Validate URL format
        if (cleanUrl.match(/^https?:\/\//i)) {
            // Double-check URL is valid using URL constructor
            try {
                new URL(cleanUrl); // This will throw if URL is invalid
                // Convert directly to HTML link (escape URL and text for safety)
                // This HTML will be inserted into the text, then the whole text will be escaped
                // But we need to mark this as "already processed" so it doesn't get double-escaped
                // Use a special marker that we'll replace after escaping
                const linkHtml = `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" class="message-link" style="color: #3b82f6; text-decoration: underline; cursor: pointer; pointer-events: auto;">${linkText}</a>`;
                
                debugLog('app.js:765', 'Link HTML created (pre-escape)', {
                    cleanUrl: cleanUrl.substring(0, 100),
                    linkText: linkText
                }, 'C');
                
                return linkHtml;
            } catch (e) {
                // Invalid URL, return original match (will be escaped later)
                return match;
            }
        }
        // Invalid URL format, return original match (will be escaped later)
        return match;
    });
    
    // Now escape the text - but we need to protect the HTML links we just created
    // Strategy: Replace HTML links with placeholders, escape, then restore
    const htmlLinkMatches = [];
    let linkPlaceholderIndex = 0;
    processedText = processedText.replace(/<a href="([^"]+)"[^>]*>([^<]+)<\/a>/g, (match, url, text) => {
        // Use a unique placeholder that won't appear in user text and won't conflict with markdown
        // Format: [LINK:index] - square brackets are safe, won't be processed as markdown link
        const placeholder = `[LINK:${linkPlaceholderIndex}]`;
        htmlLinkMatches.push({ placeholder, url, text });
        linkPlaceholderIndex++;
        return placeholder;
    });
    
    // Escape the text (HTML links are now protected as placeholders)
    let html = escapeHtml(processedText);
    
    // Restore HTML links (they're already properly formatted HTML)
    htmlLinkMatches.forEach(link => {
        const linkHtml = `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer" class="message-link" style="color: #3b82f6; text-decoration: underline; cursor: pointer; pointer-events: auto;">${escapeHtml(link.text)}</a>`;
        // Placeholder was escaped, so find the escaped version
        const escapedPlaceholder = escapeHtml(link.placeholder);
        html = html.replace(new RegExp(escapeRegex(escapedPlaceholder), 'g'), linkHtml);
    });
    
    // Bold text (**text** or __text__) - re-escape after processing
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Handle any remaining markdown links that weren't caught (fallback - should rarely happen)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
        // This is a fallback for links that weren't processed above
        // They're already escaped, so we need to decode HTML entities in the URL
        let cleanUrl = url.trim();
        // Decode HTML entities that might have been introduced by escapeHtml
        cleanUrl = cleanUrl.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#039;/g, "'");
        cleanUrl = cleanUrl.replace(/[.,;:!?\s]+$/, '');
        cleanUrl = cleanUrl.replace(/[()]+$/, '');
        
        if (cleanUrl.match(/^https?:\/\//i)) {
            try {
                new URL(cleanUrl);
                return `<a href="${escapeHtml(cleanUrl)}" target="_blank" rel="noopener noreferrer" class="message-link" style="color: #3b82f6; text-decoration: underline; cursor: pointer; pointer-events: auto;">${escapeHtml(text)}</a>`;
            } catch (e) {
                return escapeHtml(match);
            }
        }
        return escapeHtml(match);
    });
    
    // CRITICAL FIX: Convert plain URLs (not in markdown format) to clickable links
    // This handles cases where AI outputs plain URLs like "https://www.example.com"
    // Strategy: Find URLs that are NOT already inside <a> tags
    // We'll process the text before HTML links are inserted, or use a simpler approach
    
    // First, protect existing HTML links by replacing them temporarily
    const linkProtection = [];
    let protectionIndex = 0;
    html = html.replace(/<a[^>]*>([^<]*)<\/a>/gi, (match, text) => {
        const placeholder = `__LINK_PROTECT_${protectionIndex}__`;
        linkProtection.push({ placeholder, match });
        protectionIndex++;
        return placeholder;
    });
    
    // Now convert plain URLs to links
    html = html.replace(/(https?:\/\/[^\s<>"']+)/gi, (match, url) => {
        // Clean URL
        let cleanUrl = url.trim();
        cleanUrl = cleanUrl.replace(/[.,;:!?)\s]+$/, ''); // Remove trailing punctuation
        
        // Validate URL
        if (cleanUrl.match(/^https?:\/\//i)) {
            try {
                new URL(cleanUrl);
                // Extract domain name for link text
                const domain = cleanUrl.replace(/^https?:\/\/(www\.)?/, '').split('/')[0];
                const linkText = domain.length > 30 ? domain.substring(0, 30) + '...' : domain;
                return `<a href="${escapeHtml(cleanUrl)}" target="_blank" rel="noopener noreferrer" class="message-link" style="color: #3b82f6; text-decoration: underline; cursor: pointer; pointer-events: auto;">${escapeHtml(linkText)}</a>`;
            } catch (e) {
                return escapeHtml(match);
            }
        }
        return escapeHtml(match);
    });
    
    // Restore protected links
    linkProtection.forEach(({ placeholder, match }) => {
        html = html.replace(placeholder, match);
    });
    
    // Numbered lists
    html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<div class="list-item"><span class="list-number">$1.</span> $2</div>');
    
    // Bullet points
    html = html.replace(/^[-*]\s+(.+)$/gm, '<div class="list-item"><span class="list-bullet">•</span> $1</div>');
    
    // Section headers (lines ending with :)
    html = html.replace(/^(.+):\s*$/gm, '<div class="section-header">$1</div>');
    
    // Preserve line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

function formatInlineMarkdown(text) {
    return formatMessage(text);
}

function addMessage(text, type, isLoading = false) {
    console.log('[MSG] addMessage called:', { type, isLoading, textLength: text?.length || 0 });
    
    if (!chatMessages) {
        console.error('[MSG] chatMessages element not found!');
        return null;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    if (isLoading) {
        messageDiv.classList.add('loading');
    }
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (type === 'user') {
        avatar.textContent = 'U';
    } else {
        avatar.textContent = 'AI';
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = '<div class="loading"></div>';
        messageDiv.id = 'loading-message';
    } else {
        // Format markdown for bot messages, plain text for user messages
        if (type === 'bot') {
            try {
                const formatted = formatMessage(text);
                console.log('[MSG] Formatted message length:', formatted.length);
                contentDiv.innerHTML = formatted;
            } catch (e) {
                console.error('[MSG] Error formatting message:', e);
                contentDiv.textContent = text;
            }
        } else {
            const p = document.createElement('p');
            p.textContent = text;
            contentDiv.appendChild(p);
        }
        
        // Add timestamp (not for loading messages)
        if (!isLoading) {
            const timestamp = document.createElement('div');
            timestamp.className = 'message-timestamp';
            timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            timestamp.setAttribute('aria-label', `Sent at ${timestamp.textContent}`);
            contentDiv.appendChild(timestamp);
        }
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to show the start of the new message (not the end)
    // Use scrollIntoView to ensure the beginning of the message is visible
    setTimeout(() => {
        messageDiv.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start',
            inline: 'nearest'
        });
    }, 100);
    
    console.log('[MSG] Message added to DOM');
    // Return the messageDiv element or its id if it exists
    // This allows callers to check if message was added, but null is OK for non-loading messages
    return messageDiv.id ? messageDiv.id : messageDiv;
}

function removeMessage(messageId) {
    if (messageId) {
        const message = document.getElementById(messageId);
        if (message) {
            message.remove();
        }
    } else {
        const loadingMessage = document.querySelector('.message.loading');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
}

function displayProducts(products) {
    console.log('[PROD] displayProducts called with', products?.length || 0, 'products');
    console.log('[PROD] Products data:', products);
    
    if (!productsList) {
        console.error('[PROD] productsList element not found');
        return;
    }
    
    if (!Array.isArray(products)) {
        console.error('[PROD] products is not an array:', typeof products, products);
        return;
    }
    
    // Show skeleton loaders while clearing
    showProductSkeletons(products.length || 3);
    
    // Small delay to show skeleton effect
    setTimeout(() => {
        productsList.innerHTML = '';
        
        if (productCount) {
            const oldCount = parseInt(productCount.textContent) || 0;
            const newCount = products.length;
            productCount.textContent = newCount;
            
            // Animate count change
            if (oldCount !== newCount && newCount > 0) {
                productCount.classList.add('updated');
                setTimeout(() => {
                    productCount.classList.remove('updated');
                }, 500);
            }
        }
        
        if (products.length === 0) {
            console.log('[PROD] No products, showing empty state');
            productsList.innerHTML = '<div class="empty-products"><svg width="64" height="64" viewBox="0 0 24 24" fill="none"><path d="M7 18c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM1 2v2h2l3.6 7.59-1.35 2.45c-.15.28-.25.61-.25.96 0 1.1.9 2 2 2h12v-2H7.42c-.14 0-.25-.11-.25-.25l.03-.12L8.1 13h7.45c.75 0 1.41-.41 1.75-1.03L21.7 4H5.21l-.94-2H1zm16 16c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" fill="currentColor"/></svg><p>No products yet</p><span>Start a conversation to see recommendations</span></div>';
            return;
        }
        
        renderProducts(products);
    }, 300);
}

function showProductSkeletons(count) {
    if (!productsList) return;
    
    productsList.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton-product-card';
        skeleton.innerHTML = `
            <div class="skeleton skeleton-image"></div>
            <div class="skeleton skeleton-text"></div>
            <div class="skeleton skeleton-text short"></div>
            <div class="skeleton skeleton-price"></div>
        `;
        productsList.appendChild(skeleton);
    }
}

function renderProducts(products) {
    if (!productsList) return;
    
    console.log('[PROD] Rendering', products.length, 'product cards');
    products.forEach((product, index) => {
        console.log(`[PROD] Rendering product ${index + 1}:`, {
            name: product.name || product.title,
            price: product.price,
            hasImage: !!(product.image || product.image_url || product.thumbnail),
            hasUrl: !!(product.product_url || product.link || product.url || product.productUrl)
        });
        const productCard = document.createElement('div');
        productCard.className = 'product-card';
        
        const productImage = document.createElement('div');
        productImage.className = 'product-image';
        
        const img = document.createElement('img');
        // Lazy loading for images
        img.loading = 'lazy';
        img.src = product.image || product.image_url || product.thumbnail || 'https://via.placeholder.com/200x200?text=No+Image';
        img.alt = product.name || product.title || 'Product';
        // Add fade-in animation when image loads
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.3s ease';
        img.onload = function() {
            this.style.opacity = '1';
        };
        img.onerror = function() {
            this.src = 'https://via.placeholder.com/200x200?text=No+Image';
            this.style.opacity = '1';
        };
        
        productImage.appendChild(img);
        
        // Add deal badge if available (positioned absolutely on image)
        // Gracefully handle missing deal_info
        if (product.deal_info && product.deal_info.deal_badge && product.deal_info.is_deal) {
            const dealBadge = document.createElement('div');
            dealBadge.className = 'deal-badge';
            
            // Determine badge type for styling
            const badgeText = product.deal_info.deal_badge || '';
            if (badgeText.toLowerCase().includes('best price')) {
                dealBadge.classList.add('best-price');
            } else if (badgeText.toLowerCase().includes('save') || badgeText.toLowerCase().includes('%')) {
                dealBadge.classList.add('save-percent');
            } else if (product.deal_info.is_limited_time) {
                dealBadge.classList.add('limited-time');
            } else {
                dealBadge.classList.add('save-percent'); // Default
            }
            
            dealBadge.textContent = badgeText;
            dealBadge.setAttribute('aria-label', `Deal: ${badgeText}`);
            
            // Add pulse animation for limited-time deals
            if (product.deal_info.is_limited_time) {
                dealBadge.classList.add('pulse-animation');
            }
            
            productImage.appendChild(dealBadge);
        }
        
        productCard.appendChild(productImage);
        
        const productInfo = document.createElement('div');
        productInfo.className = 'product-info';
        
        const title = document.createElement('h4');
        title.className = 'product-title';
        title.textContent = product.name || product.title || 'Unknown Product';
        productInfo.appendChild(title);
        
        // Enhanced price section with deal information
        // Gracefully handle missing price
        if (product.price !== undefined && product.price !== null) {
            const priceSection = document.createElement('div');
            priceSection.className = 'price-section';
            
            // Determine actual price to display (use coupon discounted price if available)
            const displayPrice = product.coupon_info?.discounted_price || product.price;
            const originalPrice = product.price;
            const hasCoupon = product.coupon_info?.has_coupon && product.coupon_info?.discounted_price;
            
            // Show original price (strikethrough) if coupon applied or if deal shows savings
            const shouldShowOriginal = (hasCoupon && originalPrice > displayPrice) || 
                                     (product.deal_info?.savings_amount > 0 && originalPrice > displayPrice);
            
            if (shouldShowOriginal && originalPrice > displayPrice) {
                const originalPriceEl = document.createElement('p');
                originalPriceEl.className = 'price-original';
                const originalPriceValue = typeof originalPrice === 'string' ? originalPrice.replace('$', '') : originalPrice;
                originalPriceEl.textContent = `$${originalPriceValue.toFixed(2)}`;
                originalPriceEl.setAttribute('aria-label', `Original price: $${originalPriceValue.toFixed(2)}`);
                priceSection.appendChild(originalPriceEl);
            }
            
            // Current/discounted price (large, prominent)
            const currentPrice = document.createElement('p');
            currentPrice.className = 'price-current';
            const priceValue = typeof displayPrice === 'string' ? displayPrice.replace('$', '') : displayPrice;
            currentPrice.textContent = priceValue.toFixed(2);
            currentPrice.setAttribute('aria-label', `Current price: $${priceValue.toFixed(2)}`);
            
            // Change color based on whether it's a deal or has coupon
            if (product.deal_info?.is_deal || hasCoupon) {
                currentPrice.style.color = '#10b981'; // Green for deals
            } else {
                currentPrice.style.color = 'var(--accent)'; // Default accent color
            }
            
            priceSection.appendChild(currentPrice);
            
            // Savings calculation display
            let savingsDisplayed = false;
            
            // Show savings from deal if available
            if (product.deal_info?.savings_amount > 0 && product.deal_info?.savings_percent > 0) {
                const savings = document.createElement('p');
                savings.className = 'price-savings';
                const savingsAmount = product.deal_info.savings_amount.toFixed(2);
                const savingsPercent = product.deal_info.savings_percent.toFixed(0);
                savings.textContent = `Save $${savingsAmount} (${savingsPercent}% off)`;
                savings.setAttribute('aria-label', `Savings: $${savingsAmount}, ${savingsPercent} percent off`);
                priceSection.appendChild(savings);
                savingsDisplayed = true;
            }
            
            // Show coupon savings if available and not already shown
            if (hasCoupon && !savingsDisplayed) {
                const couponSavings = product.coupon_info.savings_amount || (originalPrice - displayPrice);
                if (couponSavings > 0) {
                    const savings = document.createElement('p');
                    savings.className = 'price-savings';
                    const savingsPercent = ((couponSavings / originalPrice) * 100).toFixed(0);
                    savings.textContent = `Save $${couponSavings.toFixed(2)} (${savingsPercent}% off)`;
                    savings.setAttribute('aria-label', `Coupon savings: $${couponSavings.toFixed(2)}, ${savingsPercent} percent off`);
                    priceSection.appendChild(savings);
                }
            }
            
            productInfo.appendChild(priceSection);
        }
        
        // Price comparison widget
        // Gracefully handle missing price_comparison
        if (product.price_comparison && product.price_comparison.retailer_count > 1) {
            const priceComparisonWidget = document.createElement('div');
            priceComparisonWidget.className = 'price-comparison-widget';
            
            const comparisonIcon = document.createElement('span');
            comparisonIcon.className = 'comparison-icon';
            comparisonIcon.innerHTML = '🛒';
            comparisonIcon.setAttribute('aria-hidden', 'true');
            
            const comparisonText = document.createElement('span');
            comparisonText.className = 'comparison-text';
            comparisonText.textContent = `Available at ${product.price_comparison.retailer_count} retailers`;
            
            const bestPriceText = document.createElement('span');
            bestPriceText.className = 'best-price-text';
            bestPriceText.textContent = `Best: $${product.price_comparison.best_price?.toFixed(2) || product.price.toFixed(2)}`;
            
            priceComparisonWidget.appendChild(comparisonIcon);
            priceComparisonWidget.appendChild(comparisonText);
            priceComparisonWidget.appendChild(bestPriceText);
            
            // Make it expandable for full comparison (Phase 2 enhancement)
            priceComparisonWidget.setAttribute('role', 'button');
            priceComparisonWidget.setAttribute('tabindex', '0');
            priceComparisonWidget.setAttribute('aria-label', `Price comparison: Available at ${product.price_comparison.retailer_count} retailers, best price $${product.price_comparison.best_price?.toFixed(2) || product.price.toFixed(2)}`);
            priceComparisonWidget.setAttribute('aria-expanded', 'false');
            
            // Store comparison data for expansion
            priceComparisonWidget.dataset.comparisonData = JSON.stringify(product.price_comparison);
            
            // Add click handler for expansion (will show tooltip/modal in Phase 2)
            priceComparisonWidget.addEventListener('click', function(e) {
                e.stopPropagation();
                showPriceComparisonTooltip(this, product.price_comparison);
            });
            
            priceComparisonWidget.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    showPriceComparisonTooltip(this, product.price_comparison);
                }
            });
            
            productInfo.appendChild(priceComparisonWidget);
        }
        
        // Enhanced coupon indicator with copy functionality
        // Gracefully handle missing coupon_info
        if (product.coupon_info && product.coupon_info.has_coupon) {
            const couponIndicator = document.createElement('div');
            couponIndicator.className = 'coupon-indicator';
            
            const couponIcon = document.createElement('span');
            couponIcon.className = 'coupon-icon';
            couponIcon.innerHTML = '🎟️';
            couponIcon.setAttribute('aria-hidden', 'true');
            
            const couponText = document.createElement('span');
            couponText.className = 'coupon-text';
            
            if (product.coupon_info.coupon_code) {
                couponText.innerHTML = `Coupon: <span class="coupon-code">${product.coupon_info.coupon_code}</span>`;
                couponIndicator.setAttribute('aria-label', `Coupon available: ${product.coupon_info.coupon_code}`);
                
                // Add copy button
                const copyButton = document.createElement('button');
                copyButton.className = 'coupon-copy-btn';
                copyButton.innerHTML = '📋';
                copyButton.setAttribute('aria-label', `Copy coupon code ${product.coupon_info.coupon_code}`);
                copyButton.setAttribute('title', 'Copy coupon code');
                
                copyButton.addEventListener('click', function(e) {
                    e.stopPropagation();
                    copyCouponCode(product.coupon_info.coupon_code, copyButton);
                });
                
                couponIndicator.appendChild(copyButton);
            } else {
                couponText.textContent = 'Coupon Available';
                couponIndicator.setAttribute('aria-label', 'Coupon available for this product');
            }
            
            couponIndicator.appendChild(couponIcon);
            couponIndicator.appendChild(couponText);
            
            // Make clickable to show coupon details
            couponIndicator.setAttribute('role', 'button');
            couponIndicator.setAttribute('tabindex', '0');
            couponIndicator.addEventListener('click', function(e) {
                if (!copyButton || e.target !== copyButton) {
                    showCouponTooltip(this, product.coupon_info);
                }
            });
            
            productInfo.appendChild(couponIndicator);
        }
        
        if (product.rating) {
            const rating = document.createElement('p');
            rating.className = 'product-rating';
            rating.textContent = `Rating: ${product.rating}`;
            productInfo.appendChild(rating);
        }
        
        if (product.description) {
            const desc = document.createElement('p');
            desc.className = 'product-description';
            desc.textContent = product.description;
            productInfo.appendChild(desc);
        }
        
        // Customer value score indicator (subtle, only for high scores)
        if (product.customer_value && product.customer_value.score > 0.8) {
            const valueIndicator = document.createElement('div');
            valueIndicator.className = 'customer-value-indicator';
            
            const valueIcon = document.createElement('span');
            valueIcon.className = 'value-icon';
            valueIcon.innerHTML = '⭐';
            valueIcon.setAttribute('aria-hidden', 'true');
            
            const valueText = document.createElement('span');
            valueText.className = 'value-text';
            valueText.textContent = 'Best Value';
            
            valueIndicator.appendChild(valueIcon);
            valueIndicator.appendChild(valueText);
            valueIndicator.setAttribute('title', `Customer Value Score: ${(product.customer_value.score * 100).toFixed(0)}%`);
            valueIndicator.setAttribute('aria-label', `Best Value - Customer Value Score: ${(product.customer_value.score * 100).toFixed(0)} percent`);
            
            // Add tooltip on hover
            valueIndicator.addEventListener('mouseenter', function() {
                showValueTooltip(this, product.customer_value);
            });
            
            productInfo.appendChild(valueIndicator);
        }
        
        // Add link if available
        const productLink = product.product_url || product.link || product.url || product.productUrl;
        if (productLink) {
            const link = document.createElement('a');
            link.className = 'product-link';
            link.href = productLink;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'View Product →';
            productInfo.appendChild(link);
        }
        
        productCard.appendChild(productInfo);
        productsList.appendChild(productCard);
    });
    
    // Show products panel
    if (productsPanel) {
        productsPanel.style.display = 'flex';
    }
}

function clearProducts() {
    if (!productsList) {
        return;
    }
    productsList.innerHTML = '<div class="empty-products"><svg width="64" height="64" viewBox="0 0 24 24" fill="none"><path d="M7 18c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zM1 2v2h2l3.6 7.59-1.35 2.45c-.15.28-.25.61-.25.96 0 1.1.9 2 2 2h12v-2H7.42c-.14 0-.25-.11-.25-.25l.03-.12L8.1 13h7.45c.75 0 1.41-.41 1.75-1.03L21.7 4H5.21l-.94-2H1zm16 16c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" fill="currentColor"/></svg><p>No products yet</p><span>Start a conversation to see recommendations</span></div>';
    if (productCount) {
        productCount.textContent = '0';
        productCount.classList.remove('updated');
    }
}

// Price comparison tooltip
function showPriceComparisonTooltip(element, priceComparison) {
    // Remove existing tooltip if any
    const existingTooltip = document.querySelector('.tooltip-container');
    if (existingTooltip) {
        existingTooltip.remove();
        return;
    }
    
    if (!priceComparison || !priceComparison.retailer_options || priceComparison.retailer_options.length === 0) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-container';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.setAttribute('aria-label', 'Price comparison details');
    
    const tooltipHeader = document.createElement('div');
    tooltipHeader.className = 'tooltip-header';
    tooltipHeader.innerHTML = '<h4>Price Comparison</h4><button class="tooltip-close-btn" aria-label="Close">×</button>';
    
    const tooltipContent = document.createElement('div');
    tooltipContent.className = 'tooltip-content';
    
    if (priceComparison.best_price) {
        const bestPriceText = document.createElement('div');
        bestPriceText.className = 'best-price-text';
        bestPriceText.textContent = `Best Price: $${priceComparison.best_price.toFixed(2)}`;
        tooltipContent.appendChild(bestPriceText);
    }
    
    const retailerList = document.createElement('div');
    retailerList.className = 'retailer-list';
    
    priceComparison.retailer_options.forEach((retailer, index) => {
        const retailerItem = document.createElement('div');
        retailerItem.className = 'retailer-item';
        if (index === 0) retailerItem.classList.add('best-option');
        
        const retailerName = document.createElement('div');
        retailerName.className = 'retailer-name';
        retailerName.textContent = retailer.retailer || 'Unknown';
        
        const retailerPrice = document.createElement('div');
        retailerPrice.className = 'retailer-price';
        retailerPrice.textContent = `$${retailer.price?.toFixed(2) || '0.00'}`;
        
        const retailerShipping = document.createElement('div');
        retailerShipping.className = 'retailer-shipping';
        retailerShipping.textContent = retailer.shipping_cost > 0 ? `+$${retailer.shipping_cost.toFixed(2)} shipping` : 'Free shipping';
        
        const retailerTotal = document.createElement('div');
        retailerTotal.className = 'retailer-total';
        retailerTotal.textContent = `Total: $${(retailer.total_cost || retailer.price || 0).toFixed(2)}`;
        
        retailerItem.appendChild(retailerName);
        retailerItem.appendChild(retailerPrice);
        retailerItem.appendChild(retailerShipping);
        retailerItem.appendChild(retailerTotal);
        retailerList.appendChild(retailerItem);
    });
    
    tooltipContent.appendChild(retailerList);
    tooltip.appendChild(tooltipHeader);
    tooltip.appendChild(tooltipContent);
    
    // Position tooltip near the element
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.top = `${rect.bottom + 10}px`;
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.zIndex = '10000';
    
    document.body.appendChild(tooltip);
    
    // Close button handler
    const closeBtn = tooltip.querySelector('.tooltip-close-btn');
    closeBtn.addEventListener('click', () => tooltip.remove());
    
    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', function closeOnOutsideClick(e) {
            if (!tooltip.contains(e.target) && e.target !== element) {
                tooltip.remove();
                document.removeEventListener('click', closeOnOutsideClick);
            }
        });
    }, 100);
    
    // Close on Escape key
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            tooltip.remove();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
}

// Coupon tooltip
function showCouponTooltip(element, couponInfo) {
    // Remove existing tooltip if any
    const existingTooltip = document.querySelector('.tooltip-container');
    if (existingTooltip) {
        existingTooltip.remove();
        return;
    }
    
    if (!couponInfo || !couponInfo.has_coupon) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-container';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.setAttribute('aria-label', 'Coupon details');
    
    const tooltipHeader = document.createElement('div');
    tooltipHeader.className = 'tooltip-header';
    tooltipHeader.innerHTML = '<h4>Coupon Details</h4><button class="tooltip-close-btn" aria-label="Close">×</button>';
    
    const tooltipContent = document.createElement('div');
    tooltipContent.className = 'tooltip-content';
    
    if (couponInfo.coupon_code) {
        const couponCode = document.createElement('div');
        couponCode.className = 'coupon-code-large';
        couponCode.textContent = couponInfo.coupon_code;
        tooltipContent.appendChild(couponCode);
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'coupon-copy-btn-large';
        copyBtn.textContent = 'Copy Code';
        copyBtn.addEventListener('click', () => copyCouponCode(couponInfo.coupon_code, copyBtn));
        tooltipContent.appendChild(copyBtn);
    }
    
    if (couponInfo.savings_amount > 0) {
        const savings = document.createElement('div');
        savings.className = 'coupon-savings';
        savings.textContent = `Save $${couponInfo.savings_amount.toFixed(2)} (${couponInfo.savings_percent?.toFixed(0) || 0}% off)`;
        tooltipContent.appendChild(savings);
    }
    
    if (couponInfo.description) {
        const desc = document.createElement('div');
        desc.className = 'coupon-description';
        desc.textContent = couponInfo.description;
        tooltipContent.appendChild(desc);
    }
    
    tooltip.appendChild(tooltipHeader);
    tooltip.appendChild(tooltipContent);
    
    // Position tooltip
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.top = `${rect.bottom + 10}px`;
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.zIndex = '10000';
    
    document.body.appendChild(tooltip);
    
    // Close handlers (same as price comparison)
    const closeBtn = tooltip.querySelector('.tooltip-close-btn');
    closeBtn.addEventListener('click', () => tooltip.remove());
    
    setTimeout(() => {
        document.addEventListener('click', function closeOnOutsideClick(e) {
            if (!tooltip.contains(e.target) && e.target !== element) {
                tooltip.remove();
                document.removeEventListener('click', closeOnOutsideClick);
            }
        });
    }, 100);
    
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            tooltip.remove();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
}

// Copy coupon code
function copyCouponCode(code, button) {
    if (!code) return;
    
    navigator.clipboard.writeText(code).then(() => {
        // Visual feedback
        const originalHTML = button.innerHTML;
        button.innerHTML = '✓ Copied!';
        button.style.background = '#10b981';
        button.style.color = 'white';
        
        // Show toast notification
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.textContent = `Coupon code "${code}" copied to clipboard!`;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
        
        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.style.background = '';
            button.style.color = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert(`Failed to copy coupon code. Please copy manually: ${code}`);
    });
}

// Customer value tooltip
function showValueTooltip(element, customerValue) {
    // Remove existing tooltip if any
    const existingTooltip = document.querySelector('.value-tooltip');
    if (existingTooltip) {
        existingTooltip.remove();
        return;
    }
    
    if (!customerValue || !customerValue.breakdown) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'value-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.setAttribute('aria-label', 'Customer value breakdown');
    
    const tooltipHeader = document.createElement('div');
    tooltipHeader.className = 'tooltip-header';
    tooltipHeader.innerHTML = '<h4>Value Breakdown</h4><button class="tooltip-close-btn" aria-label="Close">×</button>';
    
    const tooltipContent = document.createElement('div');
    tooltipContent.className = 'tooltip-content';
    
    const scoreInfo = document.createElement('div');
    scoreInfo.className = 'value-score-info';
    scoreInfo.innerHTML = `<strong>Overall Score:</strong> ${(customerValue.score * 100).toFixed(0)}%`;
    tooltipContent.appendChild(scoreInfo);
    
    if (customerValue.breakdown) {
        const breakdown = document.createElement('div');
        breakdown.className = 'value-breakdown';
        breakdown.innerHTML = '<strong>Factors:</strong>';
        
        const breakdownList = document.createElement('ul');
        breakdownList.className = 'breakdown-list';
        
        const factors = {
            'price_score': 'Price',
            'shipping_score': 'Shipping',
            'discount_score': 'Discount',
            'rating_score': 'Rating',
            'deal_score': 'Deal Value'
        };
        
        for (const [key, label] of Object.entries(factors)) {
            const score = customerValue.breakdown[key];
            if (score !== undefined) {
                const item = document.createElement('li');
                item.innerHTML = `${label}: ${(score * 100).toFixed(0)}%`;
                breakdownList.appendChild(item);
            }
        }
        
        breakdown.appendChild(breakdownList);
        tooltipContent.appendChild(breakdown);
    }
    
    tooltip.appendChild(tooltipHeader);
    tooltip.appendChild(tooltipContent);
    
    // Position tooltip near the element
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.top = `${rect.bottom + 10}px`;
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.zIndex = '10000';
    
    document.body.appendChild(tooltip);
    
    // Close button handler
    const closeBtn = tooltip.querySelector('.tooltip-close-btn');
    closeBtn.addEventListener('click', () => tooltip.remove());
    
    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', function closeOnOutsideClick(e) {
            if (!tooltip.contains(e.target) && e.target !== element) {
                tooltip.remove();
                document.removeEventListener('click', closeOnOutsideClick);
            }
        });
    }, 100);
    
    // Close on Escape key
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            tooltip.remove();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
    
    // Auto-close on mouse leave
    element.addEventListener('mouseleave', () => {
        setTimeout(() => {
            if (tooltip.parentElement) {
                tooltip.remove();
            }
        }, 300);
    });
}

function updateStatus(text, type = 'ready') {
    if (statusBar) {
        statusBar.textContent = text;
        
        // Update status indicator class
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator ' + type;
        }
        setTimeout(() => {
            if (statusBar) {
                statusBar.textContent = 'Ready';
            }
        }, 5000);
    }
}

// Typing indicator functions
function showTypingIndicator() {
    // Optional: Can add visual typing indicator here if needed
    // Currently the loading message handles this
    console.log('[UI] Typing indicator shown');
}

function hideTypingIndicator() {
    // Optional: Can hide typing indicator here if needed
    // Currently the loading message handles this
    console.log('[UI] Typing indicator hidden');
}

// Check API connection on load
async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            updateStatus('Connected', 'ready');
        } else {
            updateStatus('API not responding', 'error');
        }
    } catch (error) {
        updateStatus('Cannot connect to API', 'error');
        console.error('Connection check failed:', error);
    }
}

// Voice Assistant Module
class VoiceAssistant {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.isSpeaking = false;
        this.currentUtterance = null;
        
        this.initSpeechRecognition();
    }
    
    initSpeechRecognition() {
        // Check browser support
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported in this browser');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        // Configuration
        this.recognition.continuous = false; // Stop after one result
        this.recognition.interimResults = false; // Only final results
        this.recognition.lang = 'en-US'; // Language
        
        // Event handlers
        this.recognition.onstart = () => {
            this.isListening = true;
            updateStatus('Listening...');
            if (voiceButton) {
                voiceButton.classList.add('listening');
            }
        };
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            console.log('Voice input:', transcript);
            
            // Set input value and send message
            if (chatInput) {
                chatInput.value = transcript;
                sendMessage();
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            updateStatus('Voice input error');
            this.isListening = false;
            if (voiceButton) {
                voiceButton.classList.remove('listening');
            }
            
            // User-friendly error messages
            let errorMsg = 'Voice input error. ';
            switch(event.error) {
                case 'no-speech':
                    errorMsg += 'No speech detected.';
                    break;
                case 'audio-capture':
                    errorMsg += 'Microphone not accessible.';
                    break;
                case 'not-allowed':
                    errorMsg += 'Microphone permission denied.';
                    break;
                default:
                    errorMsg += 'Please try again.';
            }
            addMessage(errorMsg, 'bot');
        };
        
        this.recognition.onend = () => {
            this.isListening = false;
            if (voiceButton) {
                voiceButton.classList.remove('listening');
            }
            if (statusBar && statusBar.textContent === 'Listening...') {
                updateStatus('Ready', 'ready');
            }
        };
    }
    
    startListening() {
        if (!this.recognition) {
            addMessage('Voice input is not supported in your browser. Please use Chrome or Edge.', 'bot');
            return;
        }
        
        if (this.isListening) {
            this.stopListening();
            return;
        }
        
        try {
            this.recognition.start();
        } catch (error) {
            console.error('Failed to start recognition:', error);
            updateStatus('Ready', 'ready');
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }
    
    speak(text, options = {}) {
        if (!this.synthesis) {
            console.warn('Speech synthesis not supported');
            return;
        }
        
        // Stop any current speech
        this.stopSpeaking();
        
        // Remove markdown and clean text for speech
        const cleanText = this.cleanTextForSpeech(text);
        
        if (!cleanText) return;
        
        const utterance = new SpeechSynthesisUtterance(cleanText);
        
        // Voice options from settings or defaults
        utterance.rate = options.rate || parseFloat(localStorage.getItem('voiceRate') || '1.0');
        utterance.pitch = options.pitch || parseFloat(localStorage.getItem('voicePitch') || '1.0');
        utterance.volume = options.volume || parseFloat(localStorage.getItem('voiceVolume') || '0.8');
        
        // Try to use a natural-sounding voice
        const voices = this.synthesis.getVoices();
        const preferredVoice = voices.find(v => 
            v.name.includes('Google') || 
            v.name.includes('Natural') ||
            (v.lang.startsWith('en') && v.localService === false)
        ) || voices.find(v => v.lang.startsWith('en'));
        
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        
        utterance.onstart = () => {
            this.isSpeaking = true;
            updateStatus('Speaking...');
        };
        
        utterance.onend = () => {
            this.isSpeaking = false;
            updateStatus('Ready', 'ready');
        };
        
        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event.error);
            this.isSpeaking = false;
            updateStatus('Ready', 'ready');
        };
        
        this.currentUtterance = utterance;
        this.synthesis.speak(utterance);
    }
    
    stopSpeaking() {
        if (this.synthesis && this.isSpeaking) {
            this.synthesis.cancel();
            this.isSpeaking = false;
            updateStatus('Ready', 'ready');
        }
    }
    
    cleanTextForSpeech(text) {
        if (!text) return '';
        
        // Remove markdown formatting
        let clean = text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
            .replace(/__(.*?)__/g, '$1') // Bold
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Links
            .replace(/#{1,6}\s+/g, '') // Headers
            .replace(/\n+/g, '. ') // Newlines to periods
            .replace(/\s+/g, ' ') // Multiple spaces
            .trim();
        
        // Limit length for better UX
        if (clean.length > 500) {
            clean = clean.substring(0, 500) + '...';
        }
        
        return clean;
    }
    
    toggleVoiceInput() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }
}

// Toast Notification System
function showToast(message, type = 'success', duration = 4000) {
    let container = document.getElementById('toast-container');
    if (!container) {
        // Create container if it doesn't exist
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        container.setAttribute('aria-live', 'polite');
        container.setAttribute('aria-atomic', 'true');
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.setAttribute('role', 'alert');
    
    // Icon based on type
    let icon = '✓';
    if (type === 'error') icon = '✕';
    else if (type === 'warning') icon = '⚠';
    
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" aria-label="Close notification" onclick="this.parentElement.remove()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/>
            </svg>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 300);
    }, duration);
}

// Browser compatibility check
function checkVoiceSupport() {
    const sttSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    const ttsSupported = 'speechSynthesis' in window;
    
    return {
        speechToText: sttSupported,
        textToSpeech: ttsSupported,
        fullySupported: sttSupported && ttsSupported
    };
}

// Interactive Background Effects
function initInteractiveEffects() {
    const cursorGlow = document.getElementById('cursor-glow');
    const body = document.body;
    const geometricPatterns = document.querySelector('.geometric-patterns');
    const meshOverlay = document.querySelector('.mesh-gradient-overlay');
    const scrollLayer = document.querySelector('.scroll-effect-layer');
    
    if (!cursorGlow) return;
    
    // Enable parallax
    body.classList.add('parallax-enabled');
    
    // Cursor-responsive glow effect
    let mouseX = 0;
    let mouseY = 0;
    let glowX = 0;
    let glowY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Smooth follow with easing
        requestAnimationFrame(() => {
            glowX += (mouseX - glowX) * 0.1;
            glowY += (mouseY - glowY) * 0.1;
            
            cursorGlow.style.left = `${glowX - 300}px`;
            cursorGlow.style.top = `${glowY - 300}px`;
            cursorGlow.classList.add('active');
        });
        
        // Parallax effect for background layers
        if (geometricPatterns) {
            const parallaxX = (e.clientX / window.innerWidth - 0.5) * 20;
            const parallaxY = (e.clientY / window.innerHeight - 0.5) * 20;
            
            geometricPatterns.style.transform = `translate(${parallaxX}px, ${parallaxY}px)`;
        }
        
        if (meshOverlay) {
            const parallaxX = (e.clientX / window.innerWidth - 0.5) * 15;
            const parallaxY = (e.clientY / window.innerHeight - 0.5) * 15;
            
            meshOverlay.style.transform = `translate(${parallaxX}px, ${parallaxY}px)`;
        }
    });
    
    // Hide glow when mouse leaves
    document.addEventListener('mouseleave', () => {
        cursorGlow.classList.remove('active');
    });
    
    // Scroll-based effects
    let lastScrollY = 0;
    window.addEventListener('scroll', () => {
        const scrollY = window.scrollY;
        const scrollDelta = scrollY - lastScrollY;
        
        if (scrollLayer) {
            const scrollProgress = Math.min(scrollY / (document.documentElement.scrollHeight - window.innerHeight), 1);
            scrollLayer.style.transform = `translateY(${-scrollProgress * 100}px)`;
        }
        
        lastScrollY = scrollY;
    }, { passive: true });
    
    // Respect reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        body.classList.remove('parallax-enabled');
        cursorGlow.style.display = 'none';
    }
}

// Connection check will be called after DOM loads

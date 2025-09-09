document.addEventListener('DOMContentLoaded', () => {
    const sendDataBtn = document.getElementById('sendDataBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const statusEl = document.getElementById('status');
    const resultContainer = document.getElementById('resultContainer');
    
    let currentTabId = null;
    // Establish a persistent connection to the background script.
    const port = chrome.runtime.connect({ name: "popup" });

    // --- Core UI Update Function (No changes here) ---
    function updateUI(analysis) {
        if (!analysis || analysis.state === 'idle') {
            btnText.textContent = 'Analyze Page';
            sendDataBtn.disabled = false;
            loader.style.display = 'none';
            statusEl.innerHTML = 'Click "Analyze Page" to begin...';
            statusEl.classList.remove('error');
            resultContainer.innerHTML = '';
            return;
        }
        if (analysis.state === 'loading') {
            btnText.textContent = 'Analyzing...';
            sendDataBtn.disabled = true;
            loader.style.display = 'block';
            statusEl.classList.remove('error');
        }
        if (analysis.state === 'complete') {
            btnText.textContent = 'Analyze Again';
            sendDataBtn.disabled = false;
            loader.style.display = 'none';
            statusEl.classList.remove('error');
        }
        if (analysis.state === 'error') {
            btnText.textContent = 'Try Again';
            sendDataBtn.disabled = false;
            loader.style.display = 'none';
            statusEl.classList.add('error');
        }
        if (analysis.statusLog && analysis.statusLog.length > 0) {
            statusEl.innerHTML = analysis.statusLog.map(log => `<p>${log}</p>`).join('');
        } else {
             statusEl.innerHTML = 'Awaiting analysis...';
        }
        if (analysis.result) {
            let formattedResult = '';
            const analysisResult = analysis.result;
            if (typeof analysisResult === 'object' && analysisResult !== null) {
                for (const [key, value] of Object.entries(analysisResult)) {
                    formattedResult += `<p><strong>${key}:</strong> ${value}</p>`;
                }
            } else {
                formattedResult = '<p>Unexpected result format.</p>';
            }
            resultContainer.innerHTML = formattedResult;
        } else {
            resultContainer.innerHTML = '';
        }
    }

    // --- Event Listeners ---

    sendDataBtn.addEventListener('click', () => {
        if (currentTabId) {
            // Send messages through the established port
            port.postMessage({ type: 'resetAnalysis', tabId: currentTabId });
            port.postMessage({ type: 'startAnalysis', tabId: currentTabId });
        }
    });

    // Listen for real-time updates from the background script via the port
    port.onMessage.addListener((message) => {
        if (message.type === 'stateUpdate' && message.tabId === currentTabId) {
            updateUI(message.data);
        }
    });

    port.onDisconnect.addListener(() => {
        console.log("Port disconnected from background script.");
        // We could potentially try to reconnect here if needed, but for a popup, it's fine.
    });

    // --- Initial Load ---
    // First, get the current tab's ID. Then, get its state from storage.
    // This is still needed to show the correct state when the popup is first opened.
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0] && tabs[0].id) {
            currentTabId = tabs[0].id;
            const storageKey = currentTabId.toString();
            
            chrome.storage.local.get(storageKey, (data) => {
                updateUI(data[storageKey]);
            });
        }
    });
});
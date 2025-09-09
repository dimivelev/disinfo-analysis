let popupPort = null;
const OFFSCREEN_DOCUMENT_PATH = '/offscreen.html';

// --- Offscreen Document Management ---
async function hasOffscreenDocument() {
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });
  return existingContexts.length > 0;
}

async function setupOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    console.log('[BG] Offscreen document already exists.');
    return;
  }
  console.log('[BG] Creating offscreen document.');
  
  // ===================================================================
  // --- THE FIX IS ON THIS LINE ---
  // ===================================================================
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: ['DOM_PARSER'], // CORRECTED: Use a valid reason from the official list
    justification: 'To run a long-running fetch stream and parse its content.',
  });
  // ===================================================================
}

// --- Main Analysis Trigger ---
async function triggerAnalysis(tabId) {
  await setupOffscreenDocument();

  // Get HTML and then send to offscreen
  const injectionResults = await chrome.scripting.executeScript({
    target: { tabId: tabId },
    function: () => document.documentElement.outerHTML
  });

  if (!injectionResults || !injectionResults[0] || !injectionResults[0].result) {
    console.error('[BG] Failed to get page HTML.');
    return;
  }
  const pageHtml = injectionResults[0].result;

  const initialState = { state: 'loading', statusLog: ['Injecting script...'], result: null };
  await updateState(tabId, initialState);
  
  chrome.runtime.sendMessage({
    type: 'startAnalysisInOffscreen',
    target: 'offscreen',
    tabId: tabId,
    html: pageHtml,
  });
}

// --- State Management ---
async function updateState(tabId, data) {
    const storageKey = tabId.toString();
    await chrome.storage.local.set({ [storageKey]: data });
    if (popupPort) {
        try {
            popupPort.postMessage({ type: 'stateUpdate', tabId: tabId, data: data });
        } catch (e) {
            console.warn('[BG] Failed to post update to closed popup.');
        }
    }
}

// --- LISTENERS ---

// 1. Listen for connections from the Popup
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'popup') return;
  popupPort = port;
  console.log("[BG] Popup connected.");

  port.onMessage.addListener(async (message) => {
    if (message.type === 'startAnalysis' && message.tabId) {
      triggerAnalysis(message.tabId);
    } else if (message.type === 'resetAnalysis' && message.tabId) {
      await chrome.storage.local.remove(message.tabId.toString());
    }
  });

  port.onDisconnect.addListener(() => {
    popupPort = null;
    console.log("[BG] Popup disconnected.");
  });
});

// 2. Listen for messages from the offscreen document
chrome.runtime.onMessage.addListener(async (message) => {
    if (message.type === 'analysisUpdateFromOffscreen') {
        const { tabId, data: offscreenData } = message;
        const storageKey = tabId.toString();
        const { [storageKey]: currentAnalysis } = await chrome.storage.local.get(storageKey);
        if (!currentAnalysis) return;
        let updatedAnalysis = { ...currentAnalysis };

        if (offscreenData.type === 'update') {
            updatedAnalysis.statusLog.push(offscreenData.payload.status);
        } else if (offscreenData.type === 'result') {
            updatedAnalysis.state = 'complete';
            updatedAnalysis.result = offscreenData.payload.result.analysis_result;
            updatedAnalysis.statusLog.push('<strong>Done!</strong>');
        } else if (offscreenData.type === 'error') {
            updatedAnalysis.state = 'error';
            updatedAnalysis.statusLog.push(`API Error: ${offscreenData.payload.error}`);
        }
        await updateState(tabId, updatedAnalysis);
    } else if (message.type === 'analysisComplete') {
        console.log(`[BG] Analysis complete for tab ${message.tabId}. Offscreen doc can be closed if idle.`);
    }
});

// 3. Listen for tab closing for cleanup
chrome.tabs.onRemoved.addListener((tabId) => {
    chrome.storage.local.remove(tabId.toString());
});
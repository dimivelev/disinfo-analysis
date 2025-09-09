const apiEndpoint = 'http://127.0.0.1:8000/analyze-html/';

// This is the main analysis function, now living in the stable offscreen document
async function startAnalysis(tabId, pageHtml) {
  console.log(`[Offscreen] Received request. Starting analysis for tab: ${tabId}`);

  const postUpdateToServiceWorker = (data) => {
    chrome.runtime.sendMessage({
      type: 'analysisUpdateFromOffscreen',
      tabId: tabId,
      data: data
    });
  };

  try {
    const initialState = { state: 'loading', statusLog: ['Connecting to API...'], result: null };
    postUpdateToServiceWorker(initialState);

    const response = await fetch(apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ html: pageHtml }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status}. ${errorText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    const processBuffer = () => {
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const message = buffer.substring(0, boundary);
        buffer = buffer.substring(boundary + 2);
        
        if (message.startsWith('data:')) {
          const jsonData = message.substring(5).trim();
          if (jsonData) {
            try {
              const data = JSON.parse(jsonData);
              postUpdateToServiceWorker({ type: data.type, payload: data });
            } catch (e) {
              console.error("[Offscreen] Failed to parse JSON:", jsonData, e);
            }
          }
        }
        boundary = buffer.indexOf('\n\n');
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        console.log(`[Offscreen] Stream finished for tab ${tabId}.`);
        if (buffer.length > 0) processBuffer();
        // Signal that this specific task is done, so the offscreen document can be closed if idle.
        chrome.runtime.sendMessage({ type: 'analysisComplete', tabId: tabId });
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      processBuffer();
    }
  } catch (error) {
    console.error('[Offscreen] Critical error:', error);
    postUpdateToServiceWorker({ type: 'error', payload: { error: error.message } });
    chrome.runtime.sendMessage({ type: 'analysisComplete', tabId: tabId });
  }
}

// Listen for messages from the service worker
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'startAnalysisInOffscreen') {
    startAnalysis(message.tabId, message.html);
  }
});
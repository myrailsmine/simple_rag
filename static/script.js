let messages = document.getElementById('chat-messages');
let thinkingPanel = document.getElementById('thinking-panel');
let thinkingSteps = document.getElementById('thinking-steps');

function addMessage(content, isUser) {
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user' : 'bot'}`;
    msg.innerHTML = marked.parse(content);  // Render Markdown
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
    MathJax.typesetPromise([msg]);  // Render LaTeX
}

function showThinking() {
    thinkingPanel.classList.remove('hidden');
    thinkingSteps.innerHTML = '<li class="loading">Grok is thinking...<div class="spinner"></div></li>';
}

function updateThinking(steps) {
    thinkingSteps.innerHTML = steps.map(step => `<li>${step}</li>`).join('');
}

function uploadPDF() {
    const fileInput = document.getElementById('pdfUpload');
    const formData = new FormData();
    formData.append('pdf', fileInput.files[0]);
    fetch('/upload_pdf', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => alert(data.message || data.error));
}

function sendQuery() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) return;
    addMessage(query, true);
    document.getElementById('queryInput').value = '';
    showThinking();
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    })
    .then(res => res.json())
    .then(data => {
        updateThinking(data.thinking_steps);
        addMessage(data.final_output, false);
    })
    .catch(err => {
        updateThinking(['Error: ' + err.message]);
        addMessage('Sorry, an error occurred.', false);
    });
}

function handleKeyPress(event) {
    if (event.key === 'Enter') sendQuery();
}

addMessage('Upload a PDF and start chatting!', false);

// frontend/script.js - Updated with Dream Mode + Upload
const API_URL = '/api';
let expertNames = ['chat', 'engineering', 'science', 'medicine', 'software_dev', 
                   'religion', 'history', 'economy', 'politics', 'literature'];
// Start status polling
let pollingInterval = null;

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(updateDreamStatus, 10000); // 10s polling (optimized)
    updateDreamStatus();
}

window.addEventListener('load', startPolling);

async function fetchWithRetry(url, options, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response;
        } catch (e) {
            if (i === retries - 1) throw e;
            await new Promise(r => setTimeout(r, 1000 * (i + 1))); // Exponential-ish backoff
        }
    }
}

// ============ Dream Mode Functions ============
async function updateDreamStatus() {
    try {
        const response = await fetchWithRetry(`${API_URL}/dream/status`);
        const status = await response.json();
        
        document.getElementById('dreamStatus').innerText = status.is_active ? '🌙 Learning' : '😴 Idle';
        document.getElementById('dreamStage').innerText = `${status.current_stage}/${status.total_stages}`;
        document.getElementById('stageName').innerText = status.stage_name;
        document.getElementById('idleTime').innerText = Math.round(status.idle_time);
        document.getElementById('idleThresholdDisplay').innerText = status.idle_threshold;
        
        // Update expert progress bars
        if (status.progress) {
            const container = document.getElementById('expertProgressBars');
            container.innerHTML = '';
            for (let i = 0; i < 10; i++) {
                const percent = status.progress[i] || 0;
                container.innerHTML += `
                    <div class="expert-bar-row">
                        <span class="expert-name">${expertNames[i]}</span>
                        <div class="expert-bar-container">
                            <div class="expert-bar" style="width: ${percent}%"></div>
                        </div>
                        <span class="expert-percent">${percent}%</span>
                    </div>
                `;
            }
        }
    } catch(e) {}
}

async function toggleDreamMode() {
    const btn = document.getElementById('dreamToggle');
    const isActive = btn.innerText.includes('Pause');
    
    if (isActive) {
        await fetch(`${API_URL}/dream/stop`, { method: 'POST' });
        btn.innerText = '▶️ Start';
    } else {
        await fetch(`${API_URL}/dream/start`, { method: 'POST' });
        btn.innerText = '⏸️ Pause';
    }
}

async function setIdleThreshold() {
    const threshold = document.getElementById('idleThresholdInput').value;
    await fetch(`${API_URL}/dream/threshold`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: parseInt(threshold) })
    });
}

// ============ Knowledge Ingestion Functions ============
async function uploadFiles(files) {
    const domain = document.getElementById('domainSelect').value;
    const formData = new FormData();
    
    for (const file of files) {
        formData.append('files', file);
    }
    formData.append('domain', domain);
    
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.style.display = 'block';
    progressDiv.innerHTML = '<div class="progress-bar" style="width: 0%"></div>';
    
    try {
        const response = await fetch(`${API_URL}/ingest/upload`, {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        
        progressDiv.innerHTML = `<div class="progress-bar" style="width: 100%"></div>`;
        setTimeout(() => { progressDiv.style.display = 'none'; }, 2000);
        
        // Show results
        const historyDiv = document.getElementById('ingestionHistory');
        for (const result of results.results) {
            const status = result.success ? '✅' : '❌';
            historyDiv.insertAdjacentHTML('afterbegin', `
                <div class="ingestion-item">
                    ${status} ${result.filename}: ${result.chunks || 0} chunks, ${result.words || 0} words
                </div>
            `);
        }
        
        // Keep only last 10
        while (historyDiv.children.length > 10) {
            historyDiv.removeChild(historyDiv.lastChild);
        }
        
        addMessage(`📚 Ingested ${results.results.length} file(s)`, 'system');
        
    } catch(err) {
        console.error(err);
        addMessage('❌ Upload failed', 'system');
        progressDiv.style.display = 'none';
    }
}

// ============ Chat Functions ============
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;
    
    const words = message.split(/\s+/);
    const useRAG = document.getElementById('ragToggle').checked;
    const useStream = document.getElementById('streamToggle').checked;
    
    addMessage(words.join(' '), 'user');
    input.value = '';
    
    // Call the backend API
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: words,
                use_rag: useRAG
            })
        });
        const data = await response.json();
        addMessage(data.response, 'bot');
    } catch (e) {
        console.error(e);
        addMessage('❌ Chat failed', 'bot');
    }
    
    // Record activity for dream mode
    await fetch(`${API_URL}/dream/activity`, { method: 'POST' });
    updateDreamStatus();
}

function addMessage(text, type) {
    const chatArea = document.getElementById('chatArea');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.innerHTML = `<div class="message-content">${text}</div>`;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ============ Initialize ============
document.getElementById('sendBtn')?.addEventListener('click', sendMessage);
document.getElementById('messageInput')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

document.getElementById('dreamToggle')?.addEventListener('click', toggleDreamMode);
document.getElementById('idleThresholdInput')?.addEventListener('change', setIdleThreshold);
document.getElementById('uploadBtn')?.addEventListener('click', () => {
    const files = document.getElementById('fileInput').files;
    if (files.length > 0) uploadFiles(files);
});

// ============ Software Architect Functions ============
async function buildSoftware() {
    const projName = document.getElementById('projNameInput').value.trim();
    const requirements = document.getElementById('projRequirements').value.trim();
    
    if (!projName || !requirements) {
        alert('Please provide project name and requirements');
        return;
    }
    
    const buildOutput = document.getElementById('buildOutput');
    const statusText = document.getElementById('buildStatusText');
    const terminal = document.getElementById('buildTerminal');
    const filesList = document.getElementById('buildFilesList');
    const downloadLink = document.getElementById('downloadLink');
    const buildBtn = document.getElementById('buildBtn');
    
    buildBtn.disabled = true;
    buildOutput.style.display = 'block';
    statusText.innerText = '⚙️ Planning & Coding...';
    terminal.innerText = 'Initializing Autonomous Architect agent...\n';
    downloadLink.style.display = 'none';
    
    try {
        const response = await fetch(`${API_URL}/build`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_name: projName, requirements: requirements })
        });
        
        const result = await response.json();
        
        if (result.success) {
            statusText.innerText = '✅ Success';
            statusText.style.color = '#10b981';
            filesList.innerText = result.files.join(', ');
            terminal.innerText += `\n --- Execution Output ---\n${result.output}\n`;
            
            if (result.zip_url) {
                downloadLink.href = result.zip_url;
                downloadLink.style.display = 'block';
                downloadLink.innerText = `📥 Download ${projName}.zip`;
            }
        } else {
            statusText.innerText = `❌ Failed at ${result.stage || 'unknown'}`;
            statusText.style.color = '#ef4444';
            terminal.innerText += `\n --- Error ---\n${result.error}\n`;
            if (result.files) filesList.innerText = result.files.join(', ');
        }
    } catch (e) {
        statusText.innerText = '❌ Error';
        terminal.innerText += `\nFatal Error: ${e.message}\n`;
    } finally {
        buildBtn.disabled = false;
    }
}

// Master Dream Mode switch
document.getElementById('buildBtn')?.addEventListener('click', buildSoftware);
document.getElementById('dreamModeToggle')?.addEventListener('change', async function() {
    try {
        const endpoint = this.checked ? '/dream/start' : '/dream/stop';
        await fetch(`${API_URL}${endpoint}`, { method: 'POST' });
        startPolling();
    } catch (e) {
        console.error('Failed to toggle dream mode master switch:', e);
    }
});

// Initial updates done via window event listener load
updateDreamStatus();

/* ============================================
   ICU Dashboard — Application Logic
   ============================================ */

const API_URL = 'http://localhost:8000';
const REFRESH_INTERVAL = 10000; // 10 seconds
const NUM_PATIENTS = 20;

// --- State ---
let patients = [];
let selectedPatient = null;
let activeFilter = 'ALL';
let apiOnline = false;
let refreshTimer = null;

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initializePatients();
    setupEventListeners();
    refreshAll();
    startAutoRefresh();
    startClock();
});

function initializePatients() {
    const names = [
        'M. Laurent', 'F. Dubois', 'M. Bernard', 'F. Moreau', 'M. Petit',
        'F. Robert', 'M. Richard', 'F. Durand', 'M. Leroy', 'F. Simon',
        'M. Michel', 'F. Garcia', 'M. David', 'F. Bertrand', 'M. Roux',
        'F. Vincent', 'M. Clement', 'F. Morel', 'M. Fournier', 'F. Andre'
    ];

    for (let i = 0; i < NUM_PATIENTS; i++) {
        patients.push({
            id: i,
            subject_id: `P-${(1000 + i).toString()}`,
            stay_id: `S-${(5000 + i).toString()}`,
            name: names[i] || `Patient ${i}`,
            age: Math.floor(Math.random() * 45) + 40,
            gender: Math.random() > 0.5 ? 1 : 0,
            riskHistory: Array.from({length: 5}, () => Math.random() * 0.3 + 0.1),
            lastPrediction: null,
            vitals: generateVitals()
        });
    }
}

function generateVitals() {
    return {
        heart_rate: Array.from({length: 10}, () => Math.floor(Math.random() * 70) + 60),
        spo2: Array.from({length: 10}, () => Math.floor(Math.random() * 12) + 88),
        resp_rate: Array.from({length: 10}, () => Math.floor(Math.random() * 16) + 12),
        temp: Array.from({length: 10}, () => +(Math.random() * 3.5 + 36).toFixed(1)),
        mean_bp: Array.from({length: 10}, () => Math.floor(Math.random() * 50) + 60)
    };
}

function setupEventListeners() {
    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            activeFilter = btn.dataset.filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderPatientGrid();
        });
    });

    // Refresh button
    document.getElementById('btn-refresh').addEventListener('click', () => {
        const btn = document.getElementById('btn-refresh');
        btn.classList.add('spinning');
        refreshAll().then(() => {
            setTimeout(() => btn.classList.remove('spinning'), 600);
        });
    });

    // Close detail panel
    document.getElementById('btn-close-detail').addEventListener('click', () => {
        selectedPatient = null;
        document.getElementById('detail-panel').classList.remove('visible');
        document.querySelectorAll('.patient-card').forEach(c => c.classList.remove('selected'));
    });
}

// --- API Communication ---
async function checkApiHealth() {
    try {
        const resp = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(2000) });
        apiOnline = resp.ok;
    } catch {
        apiOnline = false;
    }
    updateStatusIndicator();
}

async function getPrediction(patient) {
    const payload = {
        subject_id: patient.subject_id,
        stay_id: patient.stay_id,
        gender: patient.gender,
        anchor_age: patient.age,
        vitals: patient.vitals
    };

    try {
        const resp = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(5000)
        });

        if (resp.ok) {
            return await resp.json();
        }
    } catch (e) {
        // Fail silently — API might be offline
    }
    return null;
}

// --- Data Refresh ---
async function refreshAll() {
    await checkApiHealth();

    const promises = patients.map(async (p) => {
        p.vitals = generateVitals(); // Simulate new vitals
        const pred = await getPrediction(p);
        if (pred) {
            p.lastPrediction = pred;
            p.riskHistory.push(pred.risk_score);
            if (p.riskHistory.length > 30) p.riskHistory.shift();
        }
    });

    await Promise.allSettled(promises);
    renderAll();
}

function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(refreshAll, REFRESH_INTERVAL);
}

// --- Rendering ---
function renderAll() {
    renderKPIs();
    renderPatientGrid();
    renderFilterCounts();
    renderAlertBanner();
    renderSidebarStats();
    if (selectedPatient) {
        renderDetailPanel(selectedPatient);
    }
}

function renderKPIs() {
    const total = patients.length;
    const high = patients.filter(p => p.lastPrediction?.risk_level === 'HIGH').length;
    const moderate = patients.filter(p => p.lastPrediction?.risk_level === 'MODERATE').length;
    const avgScore = patients.reduce((acc, p) => acc + (p.lastPrediction?.risk_score || 0), 0) / total;

    document.getElementById('kpi-census').textContent = total;
    document.getElementById('kpi-critical').textContent = high;
    document.getElementById('kpi-moderate').textContent = moderate;
    document.getElementById('kpi-avg-risk').textContent = (avgScore * 100).toFixed(1) + '%';

    // Color the critical KPI
    const critEl = document.getElementById('kpi-critical');
    critEl.style.color = high > 0 ? 'var(--risk-high)' : 'var(--text-primary)';
}

function renderPatientGrid() {
    const grid = document.getElementById('patient-grid');
    grid.innerHTML = '';

    let filtered = patients;
    if (activeFilter !== 'ALL') {
        filtered = patients.filter(p => p.lastPrediction?.risk_level === activeFilter);
    }

    filtered.forEach((p, idx) => {
        const pred = p.lastPrediction;
        if (!pred) return;

        const level = pred.risk_level.toLowerCase();
        const score = pred.risk_score;
        const lastHR = p.vitals.heart_rate[p.vitals.heart_rate.length - 1];
        const lastSpO2 = p.vitals.spo2[p.vitals.spo2.length - 1];

        const card = document.createElement('div');
        card.className = `patient-card risk-${level}${selectedPatient?.id === p.id ? ' selected' : ''}`;
        card.innerHTML = `
            <div class="card-header">
                <div>
                    <div class="card-bed">Bed ${String(idx + 1).padStart(2, '0')}</div>
                    <div class="card-id">${p.subject_id}</div>
                </div>
                <span class="risk-badge ${level}">${pred.risk_level}</span>
            </div>
            <div class="card-score">
                <span class="score-value" style="color: var(--risk-${level})">${(score * 100).toFixed(1)}</span>
                <span class="score-unit">% risk</span>
            </div>
            <div class="card-vitals">
                <div class="vital-item">
                    <span class="vital-label">HR</span>
                    <span class="vital-value">${lastHR} bpm</span>
                </div>
                <div class="vital-item">
                    <span class="vital-label">SpO2</span>
                    <span class="vital-value">${lastSpO2}%</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => {
            selectedPatient = p;
            document.querySelectorAll('.patient-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            renderDetailPanel(p);
        });

        grid.appendChild(card);
    });
}

function renderFilterCounts() {
    const counts = { ALL: patients.length, HIGH: 0, MODERATE: 0, LOW: 0 };
    patients.forEach(p => {
        const level = p.lastPrediction?.risk_level;
        if (level) counts[level]++;
    });

    Object.entries(counts).forEach(([key, val]) => {
        const el = document.getElementById(`count-${key.toLowerCase()}`);
        if (el) el.textContent = val;
    });
}

function renderAlertBanner() {
    const highCount = patients.filter(p => p.lastPrediction?.risk_level === 'HIGH').length;
    const banner = document.getElementById('alert-banner');
    const msg = document.getElementById('alert-message');

    if (highCount > 0) {
        msg.innerHTML = `<strong>${highCount} patient${highCount > 1 ? 's' : ''}</strong> detected with significant deterioration risk. Immediate clinical review recommended.`;
        banner.classList.add('visible');
    } else {
        banner.classList.remove('visible');
    }
}

function renderSidebarStats() {
    document.getElementById('stat-model').textContent = 'XGB v1.0';
    document.getElementById('stat-horizon').textContent = '6-12h';
    document.getElementById('stat-refresh').textContent = `${REFRESH_INTERVAL / 1000}s`;
}

function renderDetailPanel(patient) {
    const panel = document.getElementById('detail-panel');
    panel.classList.add('visible');

    const pred = patient.lastPrediction;
    const level = pred?.risk_level?.toLowerCase() || 'low';

    document.getElementById('detail-initials').textContent = patient.subject_id.slice(0, 2);
    document.getElementById('detail-name').textContent = patient.subject_id;
    document.getElementById('detail-meta').textContent =
        `Age ${patient.age} | ${patient.gender === 1 ? 'Male' : 'Female'} | Stay ${patient.stay_id}`;

    // Stats
    document.getElementById('detail-risk-score').textContent = `${((pred?.risk_score || 0) * 100).toFixed(1)}%`;
    document.getElementById('detail-risk-score').style.color = `var(--risk-${level})`;
    document.getElementById('detail-risk-level').textContent = pred?.risk_level || 'N/A';
    document.getElementById('detail-hr').textContent = patient.vitals.heart_rate[patient.vitals.heart_rate.length - 1] + ' bpm';
    document.getElementById('detail-spo2').textContent = patient.vitals.spo2[patient.vitals.spo2.length - 1] + '%';
    document.getElementById('detail-rr').textContent = patient.vitals.resp_rate[patient.vitals.resp_rate.length - 1] + ' /min';
    document.getElementById('detail-temp').textContent = patient.vitals.temp[patient.vitals.temp.length - 1] + ' C';

    // Render chart
    renderRiskChart(patient.riskHistory);

    // Scroll to it
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- Chart Rendering (Canvas) ---
function renderRiskChart(history) {
    const canvas = document.getElementById('risk-chart');
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    ctx.scale(2, 2);

    const w = rect.width;
    const h = rect.height;
    const padding = { top: 10, right: 20, bottom: 30, left: 40 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i;
        const val = (100 - i * 25);
        ctx.fillText(val + '%', padding.left - 6, y + 3);
    }

    if (history.length < 2) return;

    // Threshold line at 70%
    const thresholdY = padding.top + chartH * (1 - 0.7);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, thresholdY);
    ctx.lineTo(w - padding.right, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'left';
    ctx.fillText('Critical Threshold', padding.left + 4, thresholdY - 5);

    // Plot the line
    const stepX = chartW / (history.length - 1);

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');

    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartH);
    history.forEach((val, i) => {
        const x = padding.left + i * stepX;
        const y = padding.top + chartH * (1 - val);
        ctx.lineTo(x, y);
    });
    ctx.lineTo(padding.left + (history.length - 1) * stepX, padding.top + chartH);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    history.forEach((val, i) => {
        const x = padding.left + i * stepX;
        const y = padding.top + chartH * (1 - val);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Last point dot
    const lastVal = history[history.length - 1];
    const lastX = padding.left + (history.length - 1) * stepX;
    const lastY = padding.top + chartH * (1 - lastVal);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fillStyle = lastVal > 0.7 ? '#ef4444' : '#6366f1';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

// --- Status & Clock ---
function updateStatusIndicator() {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (apiOnline) {
        dot.classList.remove('offline');
        text.textContent = 'Connected';
    } else {
        dot.classList.add('offline');
        text.textContent = 'Offline';
    }
}

function startClock() {
    const el = document.getElementById('clock');
    setInterval(() => {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }, 1000);
}

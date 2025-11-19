const API_BASE = window.location.origin;

async function loadDashboard() {
    const days = document.getElementById('daysPeriod').value;
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('dashboard').style.display = 'none';

    try {
        // Load summary data
        const summaryResponse = await fetch(`${API_BASE}/api/stt/usage/summary?days=${days}`);
        const summaryData = await summaryResponse.json();

        // Load detailed data
        const detailedResponse = await fetch(`${API_BASE}/api/stt/usage/detailed?days=${days}&limit=50`);
        const detailedData = await detailedResponse.json();

        if (summaryData.success) {
            populateDashboard(summaryData.data, detailedData.data || []);
        } else {
            throw new Error('Failed to load summary data');
        }

    } catch (error) {
        console.error('Error loading dashboard:', error);
        document.getElementById('loading').innerHTML = `<p style="color: red;">Error loading data: ${error.message}</p>`;
    }
}

function populateDashboard(summary, recentRecords) {
    // Populate stats cards
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${summary.summary.total_requests}</div>
            <div class="stat-label">Total Requests</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${summary.summary.total_audio_duration_hours}h</div>
            <div class="stat-label">Audio Processed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${summary.summary.total_audio_size_mb} MB</div>
            <div class="stat-label">Data Processed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${summary.summary.success_rate_percent}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${summary.summary.average_processing_time_ms}ms</div>
            <div class="stat-label">Avg Processing Time</div>
        </div>
    `;

    // Populate endpoint table
    const endpointTableBody = document.getElementById('endpointTableBody');
    endpointTableBody.innerHTML = '';
    Object.entries(summary.endpoints).forEach(([endpoint, stats]) => {
        const successRate = ((stats.success_count / stats.requests) * 100).toFixed(1);
        const row = `
            <tr>
                <td>${endpoint}</td>
                <td>${stats.requests}</td>
                <td>${(stats.duration_seconds / 3600).toFixed(2)}</td>
                <td>${(stats.size_bytes / (1024 * 1024)).toFixed(2)}</td>
                <td class="${successRate >= 95 ? 'success' : 'error'}">${successRate}%</td>
            </tr>
        `;
        endpointTableBody.innerHTML += row;
    });

    // Populate language table
    const languageTableBody = document.getElementById('languageTableBody');
    languageTableBody.innerHTML = '';
    Object.entries(summary.languages).forEach(([language, stats]) => {
        const row = `
            <tr>
                <td>${language}</td>
                <td>${stats.requests}</td>
                <td>${(stats.duration_seconds / 3600).toFixed(2)}</td>
            </tr>
        `;
        languageTableBody.innerHTML += row;
    });

    // Populate recent activity table
    const recentTableBody = document.getElementById('recentTableBody');
    recentTableBody.innerHTML = '';
    recentRecords.slice(0, 20).forEach(record => {
        const timestamp = new Date(record.timestamp).toLocaleString('en-IN', {timeZone: 'Asia/Kolkata'});
        const status = record.success ? 'success' : 'error';
        const statusText = record.success ? 'Success' : 'Failed';
        const row = `
            <tr>
                <td>${timestamp}</td>
                <td>${record.session_id || 'N/A'}</td>
                <td>${record.endpoint}</td>
                <td>${record.audio_duration_seconds.toFixed(1)}</td>
                <td>${(record.audio_file_size_bytes / 1024).toFixed(1)}</td>
                <td>${record.language_code}</td>
                <td class="${status}">${statusText}</td>
            </tr>
        `;
        recentTableBody.innerHTML += row;
    });

    document.getElementById('loading').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
}

async function exportData(format) {
    const days = document.getElementById('daysPeriod').value;
    
    try {
        const response = await fetch(`${API_BASE}/api/stt/usage/export?days=${days}&format=${format}`);
        
        if (format === 'csv') {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `stt_usage_${days}days.csv`;
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            const data = await response.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `stt_usage_${days}days.json`;
            a.click();
            window.URL.revokeObjectURL(url);
        }
    } catch (error) {
        alert('Error exporting data: ' + error.message);
    }
}

// Load dashboard on page load
document.addEventListener('DOMContentLoaded', loadDashboard);
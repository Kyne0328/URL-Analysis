// Toast Notification System
function showToast(message, type = 'info', title = '') {
    const toastContainer = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        error: 'fa-times-circle',
        success: 'fa-check-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-triangle'
    };

    const titles = {
        error: title || 'Error',
        success: title || 'Success',
        info: title || 'Info',
        warning: title || 'Warning'
    };

    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${icons[type]}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-title">${titles[type]}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Tooltip helper function
function createTooltip(label, content) {
    return `
        <div class="tooltip-wrapper">
            <span>${label}</span>
            <span class="tooltip-trigger" tabindex="0" role="tooltip" aria-label="${content}">
                <i class="fas fa-question-circle"></i>
                <span class="tooltip-content">${content}</span>
            </span>
        </div>
    `;
}

// Extract domain from URL
function extractDomain(url) {
    try {
        const urlObj = new URL(url.startsWith('http') ? url : 'https://' + url);
        return urlObj.hostname;
    } catch {
        return url.split('/')[0];
    }
}

// Format URL with domain highlighting
function formatURLDisplay(url) {
    try {
        const urlObj = new URL(url.startsWith('http') ? url : 'https://' + url);
        const domain = urlObj.hostname;
        const path = urlObj.pathname + urlObj.search + urlObj.hash;
        return `<span class="domain-part">${domain}</span><span class="path-part">${path}</span>`;
    } catch {
        return url;
    }
}

// Get favicon letter
function getFaviconLetter(domain) {
    const cleanDomain = domain.replace('www.', '');
    return cleanDomain.charAt(0).toUpperCase();
}

// Display Results - THIS FUNCTION IS COMPLETELY REWRITTEN
function displayResults(data) {
    const urlInfo = data.url_info;
    const purityInfo = urlInfo.cluster_purity_info;

    // --- 1. Determine Result Styling based on Purity Info ---
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    // --- REWRITTEN SECTION: Remove duplicated logic and use backend's classification ---
    // Get the classification directly from the backend response
    const iconClass = urlInfo.pattern_icon;
    const iconStyle = urlInfo.pattern_style;
    const title = urlInfo.pattern_group;
    let subtitle;

    // Construct a descriptive subtitle based on the data
    if (!purityInfo || purityInfo.total_count === 0) {
        subtitle = 'This URL belongs to a cluster with no known labeled data.';
    } else {
        const phishing_percent = (purityInfo.phishing_count / purityInfo.total_count) * 100;
        const legitimate_percent = (purityInfo.legitimate_count / purityInfo.total_count) * 100;

        subtitle = `This pattern group contains ${phishing_percent.toFixed(0)}% suspicious and ${legitimate_percent.toFixed(0)}% normal URLs from our dataset.`;
    }

    resultIcon.className = `result-icon ${iconStyle}`;
    resultIcon.innerHTML = `<i class="${iconClass}"></i>`;
    resultTitle.textContent = title;
    resultSubtitle.textContent = subtitle;
    // --- END OF REWRITTEN SECTION ---

    // --- MODIFIED SECTION: Add dynamic block for suspicious keywords ---
    const resultDetails = document.getElementById('resultDetails');

    // Create a new div for the keyword warning if needed
    let keywordWarningHtml = '';
    if (urlInfo.suspicious_kw_count && urlInfo.suspicious_kw_count > 0) {
        keywordWarningHtml = `
            <div class="detail-item" style="grid-column: 1 / -1; background: rgba(245, 101, 101, 0.1); border-color: rgba(245, 101, 101, 0.3);">
                <div class="detail-label">
                    <i class="fas fa-keyboard" style="color: #f56565;"></i>
                    ${createTooltip('Heuristic Keyword Match', 'This URL contains keywords commonly found in phishing links. This increases its risk score.')}
                </div>
                <div class="detail-value" style="font-size: 16px; color: #ffbaba;">
                    Found ${urlInfo.suspicious_kw_count} suspicious keyword(s) (e.g., "login", "secure", "update").
                </div>
            </div>
        `;
    }

    // Populate the main details grid
    resultDetails.innerHTML = `
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Cluster ID', 'The group number this URL belongs to based on similar structural patterns.')}</div>
            <div class="detail-value">${urlInfo.cluster_id || 'N/A'}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Pattern Match (Confidence)', 'A 0-100% score based on the combined risk. Higher is better.')}</div>
            <div class="detail-value">${(urlInfo.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Combined Risk Score', 'Raw score from the model where higher values indicate higher risk. This is used to calculate the Pattern Match score.')}</div>
            <div class="detail-value">${urlInfo.risk_score ? urlInfo.risk_score.toFixed(3) : 'N/A'}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Neighbor Consistency', 'How consistently the nearest neighbors match the analyzed URL\'s pattern group.')}</div>
            <div class="detail-value">${(urlInfo.neighbor_confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Cluster Purity', 'The percentage of the dominant URL type (suspicious or normal) within this cluster.')}</div>
            <div class="detail-value" style="color:${purityInfo.majority_class === 'phishing' ? '#f56565' : '#48bb78'};">
                ${(purityInfo.purity * 100).toFixed(1)}% ${purityInfo.majority_class}
            </div>
        </div>
        <div class="detail-item">
            <div class="detail-label">${createTooltip('Cluster Composition', 'The breakdown of known URL types within this cluster from the training dataset.')}</div>
            <div class="detail-value" style="font-size: 16px;">
                <i class="fas fa-user-secret" style="color: #f56565;"></i> ${purityInfo.phishing_count} Phishing /
                <i class="fas fa-shield-alt" style="color: #48bb78;"></i> ${purityInfo.legitimate_count} Legitimate
            </div>
        </div>
        <!-- The new keyword warning block will be inserted here -->
        ${keywordWarningHtml}
    `;
    // --- END OF MODIFIED SECTION ---

    // Update neighbors section with improved formatting
    if (urlInfo.nearest_neighbors && urlInfo.nearest_neighbors.length > 0) {
        const neighborsSection = document.getElementById('neighborsSection');
        const neighborsGrid = document.getElementById('neighborsGrid');

        neighborsGrid.innerHTML = urlInfo.nearest_neighbors.map(nn => {
            const patternType = nn.label === 'phishing' ? 'Suspicious' :
                              nn.label === 'legitimate' ? 'Normal' : 'Unknown';
            const badgeClass = nn.label === 'phishing' ? 'suspicious' :
                             nn.label === 'legitimate' ? 'safe' : 'unknown';

            const domain = extractDomain(nn.url);
            const faviconLetter = getFaviconLetter(domain);
            const formattedURL = formatURLDisplay(nn.url);
            // Distance is now normalized (0-1), so similarity = 1 - distance
            const similarity = (1 - nn.distance).toFixed(3);
            const similarityPercent = ((1 - nn.distance) * 100).toFixed(1);

            return `
                <div class="neighbor-item" role="article" aria-label="Similar URL: ${domain}">
                    <div class="neighbor-header">
                        <div class="neighbor-favicon" aria-hidden="true">${faviconLetter}</div>
                        <div class="neighbor-domain">${domain}</div>
                        <div class="neighbor-badge ${badgeClass}">${patternType}</div>
                    </div>
                    <div class="neighbor-url">${formattedURL}</div>
                    <div class="neighbor-meta">
                        <i class="fas fa-chart-line" aria-hidden="true"></i> Similarity: ${similarityPercent}%
                        <span style="margin: 0 8px;">|</span>
                        <i class="fas fa-layer-group" aria-hidden="true"></i> Cluster: ${nn.cluster}
                    </div>
                </div>
            `;
        }).join('');

        neighborsSection.style.display = 'block';
    }

    // Update charts section with fullscreen capability
    if (data.main_dendrogram || data.purity_plot_data || data.cluster_distribution_data) {
        const chartsSection = document.getElementById('chartsSection');

        if (data.main_dendrogram) {
            document.getElementById('mainDendrogramPlot').innerHTML = `
                <img src="${data.main_dendrogram}" class="chart-image" alt="Hierarchical clustering dendrogram" onclick="openChartFullscreen(this)">
            `;
        }

        // --- NEW: Render interactive purity plot ---
        if (data.purity_plot_data) {
            renderPurityPlot(data.purity_plot_data, data.url_info);
        }

        // --- NEW: Render cluster distribution bar chart ---
        if (data.cluster_distribution_data) {
            renderClusterDistributionChart(data.cluster_distribution_data, data.url_info);
        }
        // --- NEW: Render radar chart ---
        if (data.radar_chart_data) {
            renderRadarChart(data.radar_chart_data);
        }

        chartsSection.style.display = 'grid';
    }
}

// --- NEW FUNCTION: Renders the interactive purity plot using Chart.js ---
function renderPurityPlot(plotData, urlInfo) {
    const canvas = document.getElementById('purityPlotCanvas');
    const ctx = canvas.getContext('2d');

    // Destroy previous chart instance if it exists to prevent conflicts
    if (canvas.chart) {
        canvas.chart.destroy();
    }

    const analyzedClusterId = urlInfo.cluster_id;

    // Separate the analyzed cluster's data point from the rest
    const backgroundPoints = plotData.filter(p => p.label !== `Cluster ${analyzedClusterId}`);
    const highlightedPoint = plotData.find(p => p.label === `Cluster ${analyzedClusterId}`);

    const datasets = [{
        label: 'Other Clusters',
        data: backgroundPoints,
        backgroundColor: backgroundPoints.map(p => p.majority_class === 'phishing' ? 'rgba(245, 101, 101, 0.5)' : 'rgba(72, 187, 120, 0.5)'),
        borderColor: backgroundPoints.map(p => p.majority_class === 'phishing' ? 'rgba(245, 101, 101, 0.8)' : 'rgba(72, 187, 120, 0.8)'),
        borderWidth: 1,
        pointRadius: 5,
        pointHoverRadius: 8
    }];

    if (highlightedPoint) {
        datasets.push({
            label: 'Analyzed URL\'s Cluster',
            data: [highlightedPoint],
            backgroundColor: 'rgba(102, 126, 234, 0.8)',
            borderColor: 'rgba(220, 220, 255, 1)',
            borderWidth: 2,
            pointRadius: 10,
            pointHoverRadius: 13,
            pointStyle: 'star'
        });
    }

    canvas.chart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: 'white', usePointStyle: true } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            const tooltipLines = [
                                `${point.label}`,
                                `Purity: ${point.y.toFixed(1)}% (${point.majority_class})`,
                                `Size: ${point.x.toLocaleString()} URLs`,
                                `(${point.phishing_count.toLocaleString()} phishing / ${point.legitimate_count.toLocaleString()} legit)`
                            ];
                            return tooltipLines;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Cluster Size (Number of URLs)', color: 'white' },
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Cluster Purity (%)', color: 'white' },
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// --- NEW FUNCTION: Renders the cluster distribution bar chart using Chart.js ---
function renderClusterDistributionChart(distributionData, urlInfo) {
    const canvas = document.getElementById('distributionBarCanvas');
    const ctx = canvas.getContext('2d');

    // Destroy previous chart instance if it exists to prevent conflicts
    if (canvas.chart) {
        canvas.chart.destroy();
    }

    const analyzedClusterId = urlInfo.cluster_id;

    // Prepare data for stacked bar chart
    const labels = distributionData.map(d => `Cluster ${d.cluster_id}`);
    const phishingData = distributionData.map(d => d.phishing_count);
    const legitimateData = distributionData.map(d => d.legitimate_count);

    // Find the index of the analyzed cluster
    const analyzedIndex = distributionData.findIndex(d => d.cluster_id === analyzedClusterId);

    canvas.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Suspicious URLs',
                data: phishingData,
                backgroundColor: phishingData.map((_, i) => i === analyzedIndex ? 'rgba(245, 101, 101, 1.0)' : 'rgba(245, 101, 101, 0.8)'),
                borderColor: phishingData.map((_, i) => i === analyzedIndex ? 'rgba(102, 126, 234, 1)' : 'rgba(245, 101, 101, 1)'),
                borderWidth: phishingData.map((_, i) => i === analyzedIndex ? 3 : 1),
                borderSkipped: false,
                borderRadius: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4,
                borderRadiusBottomLeft: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4,
                borderRadiusBottomRight: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4
            }, {
                label: 'Legitimate URLs',
                data: legitimateData,
                backgroundColor: legitimateData.map((_, i) => i === analyzedIndex ? 'rgba(72, 187, 120, 1.0)' : 'rgba(72, 187, 120, 0.8)'),
                borderColor: legitimateData.map((_, i) => i === analyzedIndex ? 'rgba(102, 126, 234, 1)' : 'rgba(72, 187, 120, 1)'),
                borderWidth: legitimateData.map((_, i) => i === analyzedIndex ? 3 : 1),
                borderSkipped: false,
                borderRadius: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4,
                borderRadiusTopLeft: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4,
                borderRadiusTopRight: analyzedIndex >= 0 ? labels.map((_, i) => i === analyzedIndex ? 0 : 4) : 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: 'white', usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const clusterData = distributionData[context.dataIndex];
                            const total = clusterData.total_count;
                            const purity = clusterData.purity.toFixed(1);
                            const majority = clusterData.majority_class;

                            let lines = [
                                `Total URLs: ${total.toLocaleString()}`,
                                `Purity: ${purity}% (${majority})`
                            ];

                            // Highlight the analyzed cluster
                            if (clusterData.cluster_id === analyzedClusterId) {
                                lines.push('★ This URL\'s cluster');
                            }

                            return lines;
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: { display: true, text: 'Clusters', color: 'white' },
                    ticks: {
                        color: 'white',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: { display: true, text: 'Number of URLs', color: 'white' },
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// --- NEW FUNCTION: Renders the feature comparison radar chart ---
function renderRadarChart(radarData) {
    const canvas = document.getElementById('radarChartCanvas');
    const ctx = canvas.getContext('2d');

    if (canvas.chart) {
        canvas.chart.destroy();
    }

    const data = {
        labels: radarData.labels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
        datasets: [
            {
                label: 'Analyzed URL',
                data: radarData.url_values,
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2
            },
            {
                label: 'Cluster Average',
                data: radarData.centroid_values,
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                borderColor: 'rgba(255, 255, 255, 0.7)',
                pointBackgroundColor: 'rgba(255, 255, 255, 0.7)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(255, 255, 255, 0.7)',
                borderWidth: 2
            }
        ]
    };

    canvas.chart = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: 'white' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += `${context.formattedValue}th Percentile`;
                            return label;
                        }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: 'white', font: { size: 11 } },
                    ticks: {
                        color: 'black',
                        backdropColor: 'rgba(255, 255, 255, 0.6)',
                        stepSize: 25
                    }
                }
            }
        }
    });
}

// Utility Functions
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (show) {
        spinner.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    } else {
        spinner.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze';
    }
}

function showResultsSection(show) {
    const resultsSection = document.getElementById('resultsSection');
    if (show) {
        resultsSection.style.display = 'block';
        // Add fade-in animation
        resultsSection.style.opacity = '0';
        resultsSection.style.transform = 'translateY(20px)';
        setTimeout(() => {
            resultsSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            resultsSection.style.opacity = '1';
            resultsSection.style.transform = 'translateY(0)';
        }, 10);
    } else {
        resultsSection.style.display = 'none';
    }
}

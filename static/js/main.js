document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const welcomeState = document.getElementById('welcomeState');
    const resultsDashboard = document.getElementById('resultsDashboard');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const miniPreview = document.getElementById('miniPreview');
    const resultImageMini = document.getElementById('resultImageMini');

    // Chart Instances
    let semanticChart = null;
    let compositionPieChart = null;
    let biomassHistogram = null;

    // Register Chart DataLabels
    Chart.register(ChartDataLabels);

    // --- Drag & Drop Handlers ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        document.querySelector('.upload-text').textContent = file.name;
        
        // Show mini preview
        const reader = new FileReader();
        reader.onload = function(e) {
            resultImageMini.src = e.target.result;
            miniPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // --- Form Submission ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select an image first.');
            return;
        }

        // Show Loader
        loader.classList.add('active');
        analyzeBtn.disabled = true;

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Hide Loader
            loader.classList.remove('active');
            analyzeBtn.disabled = false;

            // Render Results
            displayResults(data);

        } catch (error) {
            console.error(error);
            loader.classList.remove('active');
            analyzeBtn.disabled = false;
            alert('Error: ' + error.message);
        }
    });

    function displayResults(data) {
        // Toggle Views
        welcomeState.style.display = 'none';
        resultsDashboard.classList.add('visible');

        // Scroll to results
        resultsDashboard.scrollIntoView({ behavior: 'smooth' });

        // Update Text Metrics with Animation
        const preds = data.predictions;
        animateValue('valTotal', preds.Dry_Total_g);
        animateValue('valGDM', preds.GDM_g);
        animateValue('valGreen', preds.Dry_Green_g);
        animateValue('valClover', preds.Dry_Clover_g);
        animateValue('valDead', preds.Dry_Dead_g);

        // Update Main Image
        const imgEl = document.getElementById('resultImageMain');
        imgEl.src = 'data:image/jpeg;base64,' + data.image;

        // Update Visualization
        updatePieChart(preds);
        updateHistogram(preds);
        
        // Update Visual Analysis
        if (data.visual_analysis) {
            updateVisualAnalysis(data.visual_analysis);
        }
    }

    function updateVisualAnalysis(analysis) {
        // Update Images
        const greenImg = document.getElementById('imgGreenChannel');
        const exgImg = document.getElementById('imgExG');
        
        greenImg.src = 'data:image/jpeg;base64,' + analysis.green_channel;
        exgImg.src = 'data:image/jpeg;base64,' + analysis.exg_index;
    }

    function animateValue(id, val) {
        const el = document.getElementById(id);
        const start = 0;
        const end = parseFloat(val);
        const duration = 1000;
        let startTimestamp = null;

        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentVal = (progress * (end - start) + start);
            el.textContent = currentVal.toFixed(1);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // --- Chart Logic ---

    function updatePieChart(preds) {
        const ctx = document.getElementById('compositionPieChart').getContext('2d');
        
        // Calculate percentages
        const total = preds.Dry_Total_g || 1;
        const pGreen = ((preds.Dry_Green_g / total) * 100).toFixed(1);
        const pClover = ((preds.Dry_Clover_g / total) * 100).toFixed(1);
        const pDead = ((preds.Dry_Dead_g / total) * 100).toFixed(1);

        const data = [preds.Dry_Green_g, preds.Dry_Clover_g, preds.Dry_Dead_g];
        
        if (compositionPieChart) compositionPieChart.destroy();

        compositionPieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Dry Green', 'Dry Clover', 'Dry Dead'],
                datasets: [{
                    data: data,
                    backgroundColor: ['#4CAF50', '#9C27B0', '#795548'], // Match CSS vars
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { fontFamily: "'Open Sans', sans-serif" }
                    },
                    datalabels: {
                        color: '#fff',
                        font: { weight: 'bold' },
                        formatter: (value, ctx) => {
                            let sum = 0;
                            let dataArr = ctx.chart.data.datasets[0].data;
                            dataArr.map(data => { sum += data; });
                            let percentage = (value*100 / sum).toFixed(1)+"%";
                            return percentage;
                        }
                    },
                    tooltip: {
                         callbacks: {
                             label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)} g`
                         }
                    }
                }
            }
        });
    }

    function updateHistogram(preds) {
        // We simulate a "Histogram" / Bar Chart comparing the weights
        const ctx = document.getElementById('biomassHistogram').getContext('2d');
        
        const labels = ['Total', 'GDM', 'Green', 'Clover', 'Dead'];
        const data = [
            preds.Dry_Total_g,
            preds.GDM_g,
            preds.Dry_Green_g,
            preds.Dry_Clover_g,
            preds.Dry_Dead_g
        ];

        if (biomassHistogram) biomassHistogram.destroy();

        biomassHistogram = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Biomass (g)',
                    data: data,
                    backgroundColor: [
                        '#FF9800', // Total
                        '#8BC34A', // GDM
                        '#4CAF50', // Green
                        '#9C27B0', // Clover
                        '#795548'  // Dead
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Dry Matter (g)' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    datalabels: {
                        anchor: 'end',
                        align: 'top',
                        formatter: Math.round,
                        font: { weight: 'bold' },
                        color: '#5e6b75'
                    }
                }
            }
        });
    }
});

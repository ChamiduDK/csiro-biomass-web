document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const welcomeState = document.getElementById('welcomeState');
    const resultsDashboard = document.getElementById('resultsDashboard');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Chart Instance
    let semanticChart = null;

    // Drag & Drop Handlers
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
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        document.querySelector('.upload-text').textContent = file.name;
    }

    // Form Submission
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

        // Update Metrics
        const preds = data.predictions;
        const animateValue = (id, val) => {
            const el = document.getElementById(id);
            el.textContent = parseFloat(val).toFixed(1);
        };

        animateValue('valTotal', preds.Dry_Total_g);
        animateValue('valGreen', preds.Dry_Green_g);
        animateValue('valClover', preds.Dry_Clover_g);
        animateValue('valDead', preds.Dry_Dead_g);

        // Update Image
        const imgEl = document.getElementById('resultImage');
        imgEl.src = 'data:image/jpeg;base64,' + data.image;

        // Update Radar Chart
        updateChart(data.semantic_scores);
    }

    function updateChart(scores) {
        const ctx = document.getElementById('semanticChart').getContext('2d');
        
        // Prepare data for Radar Chart
        // We focus on the qualitative scores
        const labels = ['Green', 'Dead', 'Clover', 'Grass', 'Dense', 'Sparse', 'Bare'];
        const values = [
            scores.green, 
            scores.dead, 
            scores.clover, 
            scores.grass, 
            scores.dense,
            scores.sparse,
            scores.bare
        ];

        // Normalize for visual (simple min-max relative to group) if needed, 
        // but raw scores from SigLIP might vary. Assuming they are seemingly consistent.
        
        if (semanticChart) {
            semanticChart.destroy();
        }

        semanticChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Semantic Match Score',
                    data: values,
                    backgroundColor: 'rgba(0, 95, 75, 0.2)',
                    borderColor: '#005f4b',
                    pointBackgroundColor: '#0084c2',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#0084c2'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                family: "'Inter', sans-serif"
                            },
                            color: '#5e6b75'
                        },
                        suggestedMin: 0
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
});

# CSIRO Biomass Web App Setup Script
# This script automates the setup process for the web application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CSIRO Biomass Prediction Web App Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "`n✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Check for models directory
Write-Host "`nChecking models directory..." -ForegroundColor Yellow
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models"
    Write-Host "✓ Models directory created" -ForegroundColor Green
} else {
    Write-Host "✓ Models directory exists" -ForegroundColor Green
}

# Check if model files exist
$modelFiles = @(
    "models\ensemble_models.pkl",
    "models\feature_engine.pkl",
    "models\model_metadata.pkl"
)

$missingModels = @()
foreach ($file in $modelFiles) {
    if (-not (Test-Path $file)) {
        $missingModels += $file
    }
}

if ($missingModels.Count -gt 0) {
    Write-Host "`n⚠️  Warning: The following model files are missing:" -ForegroundColor Yellow
    foreach ($file in $missingModels) {
        Write-Host "   - $file" -ForegroundColor Yellow
    }
    Write-Host "`nPlease copy your trained model files to the models/ directory." -ForegroundColor Yellow
} else {
    Write-Host "`n✓ All required model files found" -ForegroundColor Green
}

# Create .gitignore if it doesn't exist
if (-not (Test-Path ".gitignore")) {
    Write-Host "`nCreating .gitignore..." -ForegroundColor Yellow
    @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
venv/
env/
ENV/

# Flask
instance/
.webassets-cache

# Uploads
uploads/*
!uploads/.gitkeep

# Models (optional - comment out if you want to track models)
models/*.pkl
models/siglip-*

# Environment
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Results
static/results/*
!static/results/.gitkeep
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "✓ .gitignore created" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo run the application:" -ForegroundColor Cyan
Write-Host "  1. Ensure your trained models are in the models/ directory" -ForegroundColor White
Write-Host "  2. Run: python app.py" -ForegroundColor White
Write-Host "  3. Open browser: http://localhost:5000" -ForegroundColor White
Write-Host ""

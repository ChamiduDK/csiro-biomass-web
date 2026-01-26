# PowerShell examples for CSIRO Biomass Prediction API
# Make sure the server is running: python app.py

$ApiUrl = "http://localhost:5000"

Write-Host "========================================"
Write-Host "CSIRO Biomass API - PowerShell Examples"
Write-Host "========================================"

# Health Check
Write-Host "`n1️⃣  Health Check" -ForegroundColor Cyan
Write-Host "Command: Invoke-RestMethod -Uri $ApiUrl/health"
try {
  $health = Invoke-RestMethod -Uri "$ApiUrl/health"
  $health | ConvertTo-Json
}
catch {
  Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "Make sure the server is running: python app.py" -ForegroundColor Yellow
}

# Model Information
Write-Host "`n2️⃣  Model Information" -ForegroundColor Cyan
Write-Host "Command: Invoke-RestMethod -Uri $ApiUrl/model-info"
try {
  $modelInfo = Invoke-RestMethod -Uri "$ApiUrl/model-info"
  $modelInfo | ConvertTo-Json
}
catch {
  Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Single Image Prediction
Write-Host "`n3️⃣  Single Image Prediction" -ForegroundColor Cyan
Write-Host "Example code:"
Write-Host @"
`$imagePath = "pasture.jpg"
`$form = @{
    image = Get-Item -Path `$imagePath
}
`$response = Invoke-RestMethod -Uri "$ApiUrl/predict" -Method Post -Form `$form
`$response | ConvertTo-Json -Depth 10
"@ -ForegroundColor Gray

# With specific models
Write-Host "`n4️⃣  With Specific Models" -ForegroundColor Cyan
Write-Host "Example code:"
Write-Host @"
`$imagePath = "pasture.jpg"
`$form = @{
    image = Get-Item -Path `$imagePath
    models = "lightgbm,catboost,random_forest"
}
`$response = Invoke-RestMethod -Uri "$ApiUrl/predict" -Method Post -Form `$form
`$response.predictions | Format-Table
"@ -ForegroundColor Gray

# Batch Prediction
Write-Host "`n5️⃣  Batch Prediction" -ForegroundColor Cyan
Write-Host "Example code:"
Write-Host @"
`$imageFiles = Get-ChildItem -Path "test_images" -Include *.jpg,*.png -Recurse

# Note: PowerShell native batch upload is complex
# Use Python script for batch operations:
python examples/batch_prediction.py
"@ -ForegroundColor Gray

# Save Response to File
Write-Host "`n6️⃣  Save Response to File" -ForegroundColor Cyan
Write-Host "Example code:"
Write-Host @"
`$imagePath = "pasture.jpg"
`$form = @{
    image = Get-Item -Path `$imagePath
}
`$response = Invoke-RestMethod -Uri "$ApiUrl/predict" -Method Post -Form `$form
`$response | ConvertTo-Json -Depth 10 | Out-File "result.json"
Write-Host "✅ Saved to result.json"
"@ -ForegroundColor Gray

# Extract Specific Values
Write-Host "`n7️⃣  Extract Specific Predictions" -ForegroundColor Cyan
Write-Host "Example code:"
Write-Host @"
`$imagePath = "pasture.jpg"
`$form = @{
    image = Get-Item -Path `$imagePath
}
`$response = Invoke-RestMethod -Uri "$ApiUrl/predict" -Method Post -Form `$form

if (`$response.success) {
    `$predictions = `$response.predictions
    Write-Host "Total Biomass: `$(`$predictions.Dry_Total_g) grams"
    Write-Host "Green Matter:  `$(`$predictions.Dry_Green_g) grams"
    Write-Host "Dead Matter:   `$(`$predictions.Dry_Dead_g) grams"
}
"@ -ForegroundColor Gray

# Complete Working Example
Write-Host "`n8️⃣  Complete Working Example" -ForegroundColor Cyan
Write-Host "Copy and paste this into PowerShell:"
Write-Host ""
Write-Host @"
# Complete example - change the image path
`$imagePath = "C:\path\to\your\pasture.jpg"

if (Test-Path `$imagePath) {
    `$form = @{
        image = Get-Item -Path `$imagePath
        models = "lightgbm,catboost"
    }
    
    Write-Host "📤 Sending image for prediction..."
    `$response = Invoke-RestMethod -Uri "$ApiUrl/predict" -Method Post -Form `$form
    
    if (`$response.success) {
        Write-Host "✅ Prediction successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 Results:" -ForegroundColor Cyan
        `$response.predictions.GetEnumerator() | ForEach-Object {
            Write-Host "  `$(`$_.Key): `$(`$_.Value) grams"
        }
    } else {
        Write-Host "❌ Prediction failed: `$(`$response.error)" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Image file not found: `$imagePath" -ForegroundColor Red
}
"@ -ForegroundColor Green

Write-Host "`n========================================"
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "  • Use Get-Item to properly handle file uploads"
Write-Host "  • ConvertTo-Json for pretty output"
Write-Host "  • -Depth 10 to show nested objects"
Write-Host "  • Format-Table for tabular output"
Write-Host "========================================" -ForegroundColor Yellow

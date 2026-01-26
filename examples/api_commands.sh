#!/bin/bash
# cURL examples for CSIRO Biomass Prediction API
# Make sure the server is running: python app.py

API_URL="http://localhost:5000"

echo "========================================"
echo "CSIRO Biomass API - cURL Examples"
echo "========================================"

# Health Check
echo -e "\n1️⃣  Health Check"
echo "Command: curl $API_URL/health"
curl -s $API_URL/health | jq '.' || curl -s $API_URL/health
echo ""

# Model Information
echo -e "\n2️⃣  Model Information"
echo "Command: curl $API_URL/model-info"
curl -s $API_URL/model-info | jq '.' || curl -s $API_URL/model-info
echo ""

# Single Image Prediction
echo -e "\n3️⃣  Single Image Prediction"
echo "Command: curl -X POST $API_URL/predict -F 'image=@pasture.jpg'"
echo ""
echo "📝 Note: Replace 'pasture.jpg' with your image path"
echo "Example:"
echo "  curl -X POST $API_URL/predict \\"
echo "    -F 'image=@pasture.jpg' \\"
echo "    -F 'models=lightgbm,catboost'"
echo ""

# Batch Prediction Example
echo -e "\n4️⃣  Batch Prediction"
echo "Command: curl -X POST $API_URL/batch-predict -F 'images=@img1.jpg' -F 'images=@img2.jpg'"
echo ""
echo "Example:"
echo "  curl -X POST $API_URL/batch-predict \\"
echo "    -F 'images=@image1.jpg' \\"
echo "    -F 'images=@image2.jpg' \\"
echo "    -F 'images=@image3.jpg' \\"
echo "    -F 'models=lightgbm,catboost'"
echo ""

# Save prediction to file
echo -e "\n5️⃣  Save Prediction to File"
echo "Command: curl -X POST $API_URL/predict -F 'image=@pasture.jpg' -o result.json"
echo ""
echo "Example:"
echo "  curl -X POST $API_URL/predict \\"
echo "    -F 'image=@pasture.jpg' \\"
echo "    -o result.json"
echo "  cat result.json | jq '.predictions'"
echo ""

# Pretty print with jq
echo -e "\n6️⃣  With Pretty Printing (requires jq)"
echo "Command: curl -s $API_URL/predict -F 'image=@pasture.jpg' | jq '.predictions'"
echo ""

# Multiple models
echo -e "\n7️⃣  Specify Multiple Models"
echo "Available models: lightgbm, catboost, random_forest, extra_trees, mlp, histgbm, gradient_boosting"
echo ""
echo "Example:"
echo "  curl -X POST $API_URL/predict \\"
echo "    -F 'image=@pasture.jpg' \\"
echo "    -F 'models=lightgbm,catboost,random_forest'"
echo ""

echo "========================================"
echo "💡 Tips:"
echo "  • Install jq for prettier JSON output: apt-get install jq"
echo "  • Use -s flag for silent mode (no progress)"
echo "  • Use -o to save output to file"
echo "  • Add -v for verbose output (debugging)"
echo "========================================"

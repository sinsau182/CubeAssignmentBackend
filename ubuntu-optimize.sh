#!/bin/bash
# Ubuntu Space Optimization Script for PyTorch Sentiment Analysis

echo "🐧 Ubuntu PyTorch Space Optimization"
echo "===================================="

# Activate virtual environment
source cubeenv/bin/activate

echo "🧹 Removing heavy/unnecessary packages..."

# Remove TensorFlow and related packages
pip uninstall -y tensorflow tf-keras tensorboard tensorboard-data-server || true

# Remove JAX and related packages  
pip uninstall -y jax jaxlib flax chex optax orbax-checkpoint || true

# Remove ONNX packages
pip uninstall -y onnx onnxruntime optimum || true

# Remove data processing packages (not needed for sentiment analysis)
pip uninstall -y pandas pyarrow datasets || true

# Remove image processing (not needed)
pip uninstall -y pillow opencv-python || true

# Remove other heavy ML packages
pip uninstall -y scipy matplotlib seaborn plotly bokeh || true

echo "✅ Cleanup complete!"

echo ""
echo "📦 Final package list:"
pip list | grep -E "(torch|transform|fastapi|sklearn)"

echo ""
echo "💾 Disk usage check:"
du -sh cubeenv/

echo ""
echo "🎯 Optimization complete! Your app should now use ~300MB instead of 3GB+"

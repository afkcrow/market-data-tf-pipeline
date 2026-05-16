#!/bin/bash
# setup_structure.sh - Professional MLOps folder structure for market-data-tf-pipeline
# Run this from the root of your cloned GitHub repo

echo "🚀 Creating professional MLOps project structure for market-data-tf-pipeline..."

# Create main directories
mkdir -p .github/workflows
mkdir -p src/data src/features src/models src/inference src/evaluation src/config
mkdir -p notebooks
mkdir -p tests/unit tests/integration
mkdir -p data/raw data/processed
mkdir -p models reports docs deployment
mkdir -p scripts

# Create placeholder files
touch .github/workflows/ci.yml
touch .github/workflows/deploy.yml

touch src/__init__.py
touch src/data/__init__.py src/features/__init__.py src/models/__init__.py \
      src/inference/__init__.py src/evaluation/__init__.py src/config/__init__.py

touch src/data/fetcher.py
touch src/features/technical.py
touch src/models/base_model.py
touch src/models/lstm_forecaster.py
touch src/models/trainer.py
touch src/inference/predictor.py
touch src/evaluation/backtester.py
touch src/config/settings.py

touch notebooks/01_data_exploration.ipynb
touch notebooks/02_feature_exploration.ipynb
touch notebooks/03_model_prototyping.ipynb

touch tests/unit/test_fetcher.py
touch tests/unit/test_features.py
touch tests/integration/test_pipeline.py
touch tests/conftest.py

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch reports/.gitkeep
touch docs/architecture.md

touch deployment/Dockerfile
touch .env.example
touch Makefile
touch LICENSE
touch CONTRIBUTING.md

# Create basic .gitignore (Python + data science)
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/

# Data & Models
data/processed/*
!data/processed/.gitkeep
models/*.h5
models/*.pb
*.pkl
*.joblib

# Notebooks
notebooks/*.ipynb_checkpoints/
notebooks/*/.ipynb_checkpoints/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Create basic .env.example
cat > .env.example << 'EOF'
# API Keys (never commit real keys)
# BINANCE_API_KEY=your_key_here
# BINANCE_API_SECRET=your_secret_here

# General settings
DATA_INTERVAL=15m
SYMBOLS=BTCUSDT,ETHUSDT
EOF

# Create basic pyproject.toml (using uv / modern Python)
cat > pyproject.toml << 'EOF'
[project]
name = "market-data-tf-pipeline"
version = "0.1.0"
description = "Modular CCXT + TensorFlow market data pipeline"
requires-python = ">=3.10"
dependencies = [
    "ccxt>=4.0.0",
    "tensorflow>=2.15.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pandas-ta>=0.3.14b",
    "pydantic>=2.0.0",
    "aiohttp>=3.9.0",
    "streamlit>=1.30.0",
    "pytest>=8.0.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
EOF

echo "✅ Folder structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Run: uv sync          # or pip install -e . if using pip"
echo "2. Copy your data collection logic into src/data/fetcher.py (I'll help you refactor it next)"
echo "3. Add standard features to src/features/technical.py"
echo "4. Build the README.md last"
echo ""
echo "Your repo now has the professional MLOps structure ready for refactoring!"

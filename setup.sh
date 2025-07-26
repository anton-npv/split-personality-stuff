#!/bin/bash
# Quick setup script for reasoning-faithfulness-replica
# Just run: ./setup.sh

echo "🚀 Setting up reasoning-faithfulness-replica..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install package
echo "📦 Installing dependencies..."
pip install -e .

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - API_KEY_ANTHROPIC (for Claude models)"
    echo "   - API_KEY_OPENAI (for GPT models)"
    echo "   - API_KEY_GOOGLE (for Gemini models)"
    echo "   - GROQ_API_KEY (for fast Llama inference)"
    echo ""
else
    echo "✓ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start using:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Add your API keys to .env"
echo "3. Run a quick test: make test"
echo "4. Run full experiment: make run"
echo ""
echo "Example:"
echo "  source venv/bin/activate"
echo "  make run DATASET=mmlu MODEL=claude-3-5-sonnet HINT=sycophancy"
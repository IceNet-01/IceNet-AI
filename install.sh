#!/bin/bash

# IceNet AI Installation Script for macOS
# Optimized for Apple Silicon (M1, M2, M3, M4)
# Version: 1.0.1 - Cache cleared

set -e

echo "=================================="
echo "IceNet AI Installer"
echo "Apple Silicon Optimized"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This installer is for macOS only${NC}"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This system is not Apple Silicon ($ARCH)${NC}"
    echo "IceNet is optimized for Apple Silicon, but will still work on Intel Macs"
    read -p "Continue installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}Detected macOS on $ARCH${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.10 or later from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    echo -e "${RED}Error: Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or later is required${NC}"
    exit 1
fi

echo -e "${GREEN}Python $PYTHON_VERSION found${NC}"
echo ""

# Check for git
echo "Checking for git..."
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    echo "Please install git from https://git-scm.com/downloads"
    exit 1
fi
echo -e "${GREEN}git found${NC}"
echo ""

# Check for Homebrew
echo "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew (required for AI features)..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH
    if [[ "$ARCH" == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/usr/local/bin/brew shellenv)"
    fi

    echo -e "${GREEN}✓ Homebrew installed${NC}"
else
    echo -e "${GREEN}✓ Homebrew found${NC}"
fi
echo ""

# Check for Ollama
echo "Checking for Ollama (AI model runtime)..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama for intelligent AI responses..."
    brew install ollama
    echo -e "${GREEN}✓ Ollama installed${NC}"

    # Start Ollama server in background
    echo "Starting Ollama server..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3

    # Download default model
    echo "Downloading AI model (llama3.2 - this may take a few minutes)..."
    ollama pull llama3.2:latest
    echo -e "${GREEN}✓ AI model ready!${NC}"
else
    echo -e "${GREEN}✓ Ollama found${NC}"

    # Check if ollama is running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama server..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 2
    fi

    # Check if model is installed
    if ! ollama list | grep -q "llama3.2"; then
        echo "Downloading AI model (llama3.2)..."
        ollama pull llama3.2:latest
        echo -e "${GREEN}✓ AI model ready!${NC}"
    else
        echo -e "${GREEN}✓ AI model ready!${NC}"
    fi
fi
echo ""

# Installing system-wide
echo "Installing IceNet AI system-wide..."
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install IceNet
echo ""
echo "Installing IceNet AI..."

if [ -f "setup.py" ]; then
    # Local installation (already in repository)
    echo "Installing from local source..."
    pip install -e .
else
    # Remote installation - clone from GitHub
    echo "Cloning IceNet AI from GitHub..."
    INSTALL_DIR="$HOME/.icenet-source"

    if [ -d "$INSTALL_DIR" ]; then
        echo "Removing existing installation directory..."
        rm -rf "$INSTALL_DIR"
    fi

    git clone https://github.com/IceNet-01/IceNet-AI.git "$INSTALL_DIR"

    echo "Installing from GitHub source..."
    pip install "$INSTALL_DIR"

    echo "Cleaning up..."
    rm -rf "$INSTALL_DIR"
fi

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -c "import icenet" &> /dev/null; then
    echo -e "${GREEN}IceNet AI installed successfully!${NC}"
else
    echo -e "${RED}Installation verification failed${NC}"
    exit 1
fi

# Create default directories
echo ""
echo "Creating default directories..."
mkdir -p ~/icenet/{configs,checkpoints,data,logs}
echo -e "${GREEN}Directories created at ~/icenet/${NC}"

# Generate sample config
echo ""
echo "Generating sample configuration..."
python3 -c "from icenet.core.config import Config; Config().to_yaml('$HOME/icenet/configs/example.yaml')"
echo -e "${GREEN}Sample config created at ~/icenet/configs/example.yaml${NC}"

# Display system info
echo ""
echo "=================================="
echo "System Information"
echo "=================================="
python3 -c "
from icenet.core.device import DeviceManager
dm = DeviceManager()
print(dm)
"

echo ""
echo "=================================="
echo "🎉 Installation Complete!"
echo "=================================="
echo ""
echo "Quick Start - Super Easy Mode:"
echo ""
echo "  1. Train on YOUR files (no tech skills needed!):"
echo "     $ icenet train-local ~/Documents"
echo ""
echo "  2. Chat with intelligent AI:"
echo "     $ icenet chat"
echo ""
echo "  3. Launch interactive GUI:"
echo "     $ icenet"
echo ""
echo "Advanced:"
echo "  • Setup Ollama (if skipped): icenet setup-ollama"
echo "  • Fine-tune custom model: icenet fine-tune"
echo "  • Get help: icenet --help"
echo ""
echo "Documentation: https://github.com/IceNet-01/IceNet-AI"
echo ""

# Check if Ollama was installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✨ You're all set! AI features are ready.${NC}"
    echo -e "${GREEN}   Chat will use intelligent Ollama responses!${NC}"
else
    echo -e "${YELLOW}⚠️  Install Ollama later for smart AI responses:${NC}"
    echo -e "${YELLOW}   icenet setup-ollama${NC}"
fi

echo ""
echo -e "${GREEN}Happy training with IceNet AI!${NC}"

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

# Check if virtual environment should be created
echo "Installation options:"
echo "1. Install in virtual environment (recommended)"
echo "2. Install system-wide"
read -p "Choose option (1 or 2): " -n 1 -r
#echo ""

if [[ $REPLY == "1" ]]; then
    VENV_PATH="$HOME/.icenet-venv"

    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"

    # Activate virtual environment
    source "$VENV_PATH/bin/activate"

    echo -e "${GREEN}Virtual environment created and activated${NC}"
    echo ""

    # Add activation to shell profile
    SHELL_PROFILE=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_PROFILE" ]; then
        echo "Do you want to add IceNet activation to $SHELL_PROFILE?"
        read -p "(y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "" >> "$SHELL_PROFILE"
            echo "# IceNet AI" >> "$SHELL_PROFILE"
            echo "alias icenet-activate='source $VENV_PATH/bin/activate'" >> "$SHELL_PROFILE"
            echo -e "${GREEN}Added 'icenet-activate' alias to $SHELL_PROFILE${NC}"
            echo "Run 'icenet-activate' in new terminal sessions to activate IceNet"
        fi
    fi
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install IceNet
echo ""
echo "Installing IceNet AI..."

if [ -f "setup.py" ]; then
    # Local installation (for development)
    echo "Installing from local source..."
    pip install -e .
else
    # Remote installation (for end users)
    echo "Installing from PyPI..."
    pip install icenet-ai
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
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Quick Start:"
echo "  1. Launch interactive mode:"
echo "     $ icenet"
echo ""
echo "  2. Generate a config:"
echo "     $ icenet config -o my_config.yaml"
echo ""
echo "  3. Train a model:"
echo "     $ icenet train -c my_config.yaml"
echo ""
echo "  4. Get help:"
echo "     $ icenet --help"
echo ""
echo "Documentation: https://github.com/IceNet-01/IceNet-AI"
echo ""
echo -e "${GREEN}Happy training!${NC}"

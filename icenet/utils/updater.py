"""
Auto-updater for IceNet
"""

import requests
import subprocess
import sys
from pathlib import Path
from typing import Tuple
import logging

from icenet import __version__


logger = logging.getLogger(__name__)

GITHUB_REPO = "IceNet-01/IceNet-AI"
UPDATE_CHECK_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_current_version() -> str:
    """Get current IceNet version"""
    return __version__


def check_for_updates() -> Tuple[bool, str]:
    """
    Check if updates are available

    Returns:
        Tuple of (has_update, latest_version)
    """
    try:
        response = requests.get(UPDATE_CHECK_URL, timeout=5)
        response.raise_for_status()

        data = response.json()
        latest_version = data.get("tag_name", "").lstrip("v")
        current_version = get_current_version()

        # Simple version comparison (can be improved)
        has_update = latest_version > current_version

        return has_update, latest_version

    except requests.RequestException as e:
        logger.warning(f"Failed to check for updates: {e}")
        return False, ""


def install_update() -> bool:
    """
    Install latest update using pip

    Returns:
        True if successful, False otherwise
    """
    try:
        # Upgrade via pip
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "icenet-ai",
        ])

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install update: {e}")
        return False


def auto_update_check():
    """
    Automatically check for updates on startup

    This can be called from the main CLI or UI
    """
    has_update, latest_version = check_for_updates()

    if has_update:
        logger.info(f"Update available: {latest_version}")
        logger.info("Run 'icenet update' to install")


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file from URL

    Args:
        url: URL to download from
        output_path: Path to save file

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        return False

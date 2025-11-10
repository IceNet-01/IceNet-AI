from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="icenet-ai",
    version="0.1.0",
    author="IceNet AI Team",
    author_email="info@icenet-ai.dev",
    description="A powerful AI system optimized for Apple M4 Pro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IceNet-01/IceNet-AI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "icenet=icenet.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "icenet": ["configs/*.yaml", "assets/*"],
    },
)

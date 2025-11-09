"""
Setup script for colorization package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="colorful-image-colorization",
    version="1.0.0",
    author="Samay Mehar",
    author_email="samay.m2504@gmail.com",
    description="Production-ready implementation of 'Colorful Image Colorization' (Zhang et al., ECCV 2016)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/colorization",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "isort>=5.13.2",
            "mypy>=1.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "colorize=src.infer:colorize_cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

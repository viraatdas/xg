#!/usr/bin/env python3
"""
Setup script for XG Language
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

version = "0.1.0"

setup(
    name="xg-lang",
    version=version,
    description="XG Language - A high-performance language for GPU cluster computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="XG Language Team",
    author_email="contact@xg-lang.org",
    url="https://github.com/viraatdas/xg",
    project_urls={
        "Bug Reports": "https://github.com/viraatdas/xg/issues",
        "Source": "https://github.com/viraatdas/xg",
        "Documentation": "https://github.com/viraatdas/xg#readme",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "cuda": [
            "torch[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xgc=xg.cli:xgc_main",
            "xgrun=xg.cli:xgrun_main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="gpu computing, distributed computing, tensor operations, compiler, language",
    include_package_data=True,
    zip_safe=False,
)

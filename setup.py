"""
Setup script for gradioSearch package
Created with assistance from aider.chat
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name="gradioSearch",
    version="0.1.0",
    description="A CLI tool for searching FAISS vector databases with Gradio GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TODO",
    author_email="TODO",
    url="TODO",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gradio==5.50.0",
        "pandas",
        "langchain-community",
        "sentence-transformers",
    ],
    entry_points={
        "console_scripts": [
            "gradioSearch=gradioSearch.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

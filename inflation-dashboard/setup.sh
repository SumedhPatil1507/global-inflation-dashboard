#!/bin/bash
# Install CPU-only PyTorch wheel before requirements.txt runs
# This prevents pip from pulling the 2GB GPU build or compiling from source
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu --quiet

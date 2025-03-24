#!/bin/bash

# Remove existing environment if present
rm -rf venv/

# Create fresh environment
python -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

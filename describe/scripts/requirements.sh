#!/usr/bin/env bash

# Abort if not executed as root.
if [[ $(id -u) != "0" ]]; then
  echo "Usage: run ${0##*/} as root" 1>&2
  exit 1
fi

set -o errexit

# Create virtual environment.
python3 -m venv ./venv
. ./venv/bin/activate

# Upgrade pip package manager.
./venv/bin/pip install --disable-pip-version-check --no-cache-dir --upgrade pip

# Install Python dependencies.
./venv/bin/pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

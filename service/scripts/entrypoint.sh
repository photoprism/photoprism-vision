#!/bin/bash

/app/scripts/requirements.sh

. ./venv/bin/activate

gunicorn "$@"

#!/bin/bash

/app/scripts/requirements.sh

. ./venv/bin/activate

if [ ! -z "$PHOTOPRISM_UID" ]; then
  echo "Switching to user id $PHOTOPRISM_UID..."
  exec gosu $PHOTOPRISM_UID gunicorn "$@"
else
  # Run as default user
  exec gunicorn "$@"
fi


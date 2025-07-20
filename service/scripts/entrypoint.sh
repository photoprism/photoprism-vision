#!/bin/bash

/app/scripts/requirements.sh

. ./venv/bin/activate

if [[ ! -z "$PHOTOPRISM_UID" && $(id -u) -eq 0 ]]; then
  echo "Switching from $(whoami) ($(id -u)) to user id $PHOTOPRISM_UID..."
  exec gosu $PHOTOPRISM_UID gunicorn "$@"
else
  echo "Running as default user $(whoami) ($(id -u))..."
  exec gunicorn "$@"
fi


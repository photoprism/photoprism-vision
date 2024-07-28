#!/usr/bin/env bash

# INITIALIZES CONTAINER PACKAGES AND PERMISSIONS
export PATH="/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:/app/scripts:/root/.cargo/bin:/app/venv/bin"

# Abort if not executed as root.
if [[ $(id -u) != "0" ]]; then
  echo "Usage: run ${0##*/} as root" 1>&2
  exit 1
fi

# regular expressions
re='^[0-9]+$'

# detect environment
case $DOCKER_ENV in
  prod)
    INIT_SCRIPTS="/app/scripts"
    ;;

  develop)
    INIT_SCRIPTS="/app/scripts"
    ;;

  *)
    echo "init: unsupported environment $DOCKER_ENV";
    exit
    ;;
esac

if [[ ${PHOTOPRISM_UID} =~ $re ]] && [[ ${PHOTOPRISM_UID} != "0" ]]; then
  # Create user account if it does not exist yet (required by /usr/bin/setpriv).
  getent passwd "${PHOTOPRISM_UID}" > /dev/null
  if [ $? -eq 2 ] ; then
    userdel -r -f "user-${PHOTOPRISM_UID}" >/dev/null 2>&1
    groupdel -f "group-${PHOTOPRISM_UID}" >/dev/null 2>&1
    groupadd -f -g "${PHOTOPRISM_UID}" "group-${PHOTOPRISM_UID}"
    useradd -u "${PHOTOPRISM_UID}" -g "${PHOTOPRISM_UID}" -G video,renderd,render,videodriver -s /bin/bash -m -d "/home/user-${PHOTOPRISM_UID}" "user-${PHOTOPRISM_UID}" 2>/dev/null
    echo "init: account with the user id ${PHOTOPRISM_UID} has been created"
  else
    echo "init: account with the user id ${PHOTOPRISM_UID} already exists"
  fi
fi

# do nothing if PHOTOPRISM_INIT was not set
if [[ -z ${PHOTOPRISM_INIT} ]]; then
  if [[ ${PHOTOPRISM_DEFAULT_TLS} = "true" ]]; then
    make --no-print-directory -C "$INIT_SCRIPTS" "https"
  fi
  exit
fi

INIT_LOCK="/app/scripts/.init-lock"

# execute targets via /usr/bin/make
if [[ ! -e ${INIT_LOCK} ]]; then
  for INIT_TARGET in $PHOTOPRISM_INIT; do
    echo "init: $INIT_TARGET"
    make --no-print-directory -C "$INIT_SCRIPTS" "$INIT_TARGET"
  done

  echo 1 >${INIT_LOCK}
fi

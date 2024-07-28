#!/usr/bin/env bash

# regular expressions
re='^[0-9]+$'

# set env defaults
export DOCKER_ARCH=${DOCKER_ARCH:-arch}
export DOCKER_ENV=${DOCKER_ENV:-unknown}
export DOCKER_TAG=${DOCKER_TAG:-unknown}
export PATH="/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:/app/scripts:/root/.cargo/bin:/app/venv/bin"

# detect environment
case $DOCKER_ENV in
  prod)
    INIT_SCRIPT="/app/scripts/entrypoint-init.sh";
    ;;

  develop)
    INIT_SCRIPT="/app/scripts/entrypoint-init.sh";
    ;;

  *)
    echo "entrypoint: unknown environment $DOCKER_ENV";
    INIT_SCRIPT=""
    ;;
esac

# normalize user and group ID environment variables
if [[ -z ${PHOTOPRISM_UID} ]]; then
  if [[ ${UID} =~ $re ]] && [[ ${UID} != "0" ]]; then
    export PHOTOPRISM_UID=${UID}
  elif [[ ${PUID} =~ $re ]] && [[ ${PUID} != "0" ]]; then
    export PHOTOPRISM_UID=${PUID}
  fi
  if [[ -z ${PHOTOPRISM_GID} ]]; then
    if [[ ${GID} =~ $re ]] && [[ ${GID} != "0" ]]; then
      export PHOTOPRISM_GID=${GID}
    elif [[ ${PGID} =~ $re ]] && [[ ${PGID} != "0" ]]; then
      export PHOTOPRISM_GID=${PGID}
    fi
  fi
fi

# initialize container packages and permissions
if [[ ${INIT_SCRIPT} ]] && [[ -f "${INIT_SCRIPT}" ]]; then
  if [[ $(/usr/bin/id -u) == "0" ]]; then
    echo "started $DOCKER_TAG as root ($DOCKER_ARCH-$DOCKER_ENV)"
    /bin/bash -c "${INIT_SCRIPT}"
  else
    echo "started $DOCKER_TAG as uid $(/usr/bin/id -u) ($DOCKER_ARCH-$DOCKER_ENV)"
    /usr/bin/sudo -E "${INIT_SCRIPT}"
  fi
else
  echo "started $DOCKER_TAG as uid $(/usr/bin/id -u) without init script ($DOCKER_ARCH-$DOCKER_ENV)"
fi

# display documentation info and link
if [[ $DOCKER_ENV == "prod" ]]; then
    echo "Problems? Our Troubleshooting Checklists help you quickly diagnose and solve them:";
    echo "https://docs.photoprism.app/getting-started/troubleshooting/";
fi

# set explicit home directory
export HOME="/app"

# check for alternate umask variable
if [[ -z ${PHOTOPRISM_UMASK} ]] && [[ ${UMASK} =~ $re ]] && [[ ${#UMASK} == 4 ]]; then
  export PHOTOPRISM_UMASK=${UMASK}
fi

# set file-creation mode (umask)
if [[ ${PHOTOPRISM_UMASK} =~ $re ]] && [[ ${#PHOTOPRISM_UMASK} == 4 ]]; then
  umask "${PHOTOPRISM_UMASK}"
else
  umask 0002
fi

# display additional container info for troubleshooting
echo "file umask....: \"$(umask)\" ($(umask -S))"
echo "home directory: ${HOME}"

# error code of the last executed command
ret=0

# change to another user and group on request
if [[ ${INIT_SCRIPT} ]] && [[ $(/usr/bin/id -u) == "0" ]] && [[ ${PHOTOPRISM_UID} =~ $re ]] && [[ ${PHOTOPRISM_UID} != "0" ]]; then
  # check uid and gid env variables
  if [[ ${PHOTOPRISM_GID} =~ $re ]] && [[ ${PHOTOPRISM_GID} != "0" ]]; then
    echo "switching to uid ${PHOTOPRISM_UID}:${PHOTOPRISM_GID}"
    echo "${@}"

    # run command as uid:gid
    (while /usr/bin/setpriv --reuid "${PHOTOPRISM_UID}" --regid "${PHOTOPRISM_GID}" --init-groups --inh-caps -all "$@"; ret=$?; [[ $ret -eq 0 ]]; do echo "${@}"; done) &
  else
    echo "switching to uid ${PHOTOPRISM_UID}"
    echo "${@}"

    # run command as uid
    (while /usr/bin/setpriv --reuid "${PHOTOPRISM_UID}" --regid "${PHOTOPRISM_GID}" --init-groups --inh-caps -all "$@"; ret=$?; [[ $ret -eq 0 ]]; do echo "${@}"; done) &
  fi
else
  echo "running as uid $(id -u)"
  echo "${@}"

  # run command
  (while "$@"; ret=$?; [[ $ret -eq 0 ]]; do echo "${@}"; done) &
fi

PID=$!

trap "kill -USR1 $PID" INT TERM
wait

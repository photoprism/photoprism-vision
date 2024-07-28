#!/usr/bin/env bash

# Abort if not executed as root.
if [[ $(id -u) != "0" ]]; then
  echo "Usage: run ${0##*/} as root" 1>&2
  exit 1
fi

set -o errexit

SYSTEM_ARCH=$(uname -m)
DESTARCH=${BUILD_ARCH:-$SYSTEM_ARCH}

case $DESTARCH in
  amd64 | AMD64 | x86_64 | x86-64)
    export HDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/hdf5"
    ;;

  arm64 | ARM64 | aarch64)
    export HDF5_LIBDIR="/usr/lib/aarch64-linux-gnu/hdf5"
    ;;

  arm | ARM | aarch | armv7l | armhf)
    export HDF5_LIBDIR="/usr/lib/arm-linux-gnueabihf/hdf5"
    ;;

  *)
    echo "Unsupported Machine Architecture: \"$DESTARCH\"" 1>&2
    exit 1
    ;;
esac

python3 -m venv /app/venv
. /app/venv/bin/activate
/app/venv/bin/pip install --disable-pip-version-check --no-cache-dir --upgrade pip
/app/venv/bin/pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

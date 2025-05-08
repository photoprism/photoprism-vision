#### Python Base Image <https://hub.docker.com/_/python>
FROM python:3.12-slim

# Copyright © 2024 PhotoPrism UG. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”):
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# Questions? Email us at hello@photoprism.app or visit our website to learn
# more about our team, products and services: https://www.photoprism.app/

# Set Open Container Initiative (OCI) annotations.
# See: https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.title="PhotoPrism® Vision"
LABEL org.opencontainers.image.description="Computer Vision API"
LABEL org.opencontainers.image.url="https://hub.docker.com/r/photoprism/vision"
LABEL org.opencontainers.image.source="https://github.com/photoprism/photoprism-vision/"
LABEL org.opencontainers.image.documentation="https://github.com/photoprism/photoprism-vision/blob/develop/README.md"
LABEL org.opencontainers.image.authors="PhotoPrism UG <hello@photoprism.app>"
LABEL org.opencontainers.image.vendor="PhotoPrism UG"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Declare build parameters.
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_TAG
ARG PHOTOPRISM_USER=photoprism
ARG PHOTOPRISM_UID=1000
ARG PHOTOPRISM_GID=1000

ENV PHOTOPRISM_UID=$PHOTOPRISM_UID
ENV PHOTOPRISM_GID=$PHOTOPRISM_GID

WORKDIR /app

## Create non-root user
RUN groupadd -g ${PHOTOPRISM_GID} ${PHOTOPRISM_USER} && \
    useradd -r -g ${PHOTOPRISM_GID} -u ${PHOTOPRISM_UID} -m ${PHOTOPRISM_USER}

# Set environment variables.
ENV PATH="/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:/app/scripts:/root/.cargo/bin:/app/venv/bin" \
  HDF5_PKGCONFIG_NAME="hdf5" \
  DOCKER_ARCH="${TARGETARCH}" \
  DOCKER_PLATFORM="${TARGETPLATFORM}" \
  DOCKER_TAG="${BUILD_TAG}" \
  DOCKER_ENV="prod" \
  PS1="\u@${DOCKER_TAG}:\w\$ " \
  LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
  DEBIAN_FRONTEND="noninteractive" \
  TMPDIR="/tmp" \
  TF_CPP_MIN_LOG_LEVEL="2" \
  MALLOC_ARENA_MAX="4" \
  PROG="gunicorn"


COPY --chown=root:root ./scripts ./scripts

# Install system dependencies.
RUN echo 'APT::Acquire::Retries "3";' > /etc/apt/apt.conf.d/80retries && \
  echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/80recommends && \
  echo 'APT::Install-Suggests "false";' > /etc/apt/apt.conf.d/80suggests && \
  echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/80forceyes && \
  echo 'APT::Get::Fix-Missing "true";' > /etc/apt/apt.conf.d/80fixmissing && \
  echo 'force-confold' > /etc/dpkg/dpkg.cfg.d/force-confold && \
  apt-get update && \
  apt-get -qq dist-upgrade && \
  apt-get install gosu && \
  echo 'alias ll="ls -alh"' >> /etc/skel/.bashrc && \
  echo 'export PS1="📷 \u@${BUILD_TAG}:\w\$ "' >> /etc/skel/.bashrc && \
  chmod u=rx,go=rx /app/scripts/*.sh && \
  cp /etc/skel/.bashrc /home/${PHOTOPRISM_USER}/.bashrc && \
  cleanup.sh

# Initialize dependencies
COPY --chown=root:root requirements.txt .
COPY --chown=root:root *.py ./

# Folder for local AI models
VOLUME --chown=${PHOTOPRISM_UID}:${PHOTOPRISM_GID} /app/models
# Folder for python environment. Expected size is ~ 5 GiB
VOLUME --chown=${PHOTOPRISM_UID}:${PHOTOPRISM_GID} /app/venv

# Expose HTTP service port.
EXPOSE 5000

# Set container entrypoint script.
ENTRYPOINT ["entrypoint.sh"]

# Listen at 0.0.0.0:5000 when the container is started.
CMD [ "--bind", "0.0.0.0:5000", "app:app"]

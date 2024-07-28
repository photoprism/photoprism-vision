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
LABEL org.opencontainers.image.description="Describe Service"
LABEL org.opencontainers.image.url="https://hub.docker.com/r/photoprism/vision-describe"
LABEL org.opencontainers.image.source="https://github.com/photoprism/photoprism-vision/"
LABEL org.opencontainers.image.documentation="https://github.com/photoprism/photoprism-vision/blob/develop/README.md"
LABEL org.opencontainers.image.authors="PhotoPrism UG <hello@photoprism.app>"
LABEL org.opencontainers.image.vendor="PhotoPrism UG"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Declare build parameters.
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_TAG

# Set environment variables.
ENV PATH="/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:/app/scripts:/root/.cargo/bin:/app/venv/bin" \
    HDF5_PKGCONFIG_NAME="hdf5" \
    DOCKER_TAG="${BUILD_TAG}" \
    DOCKER_ENV="prod" \
    PS1="\u@${DOCKER_TAG}:\w\$ " \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    NODE_ENV="production" \
    DEBIAN_FRONTEND="noninteractive" \
    TMPDIR="/tmp" \
    TF_CPP_MIN_LOG_LEVEL="2" \
    MALLOC_ARENA_MAX="4" \
    GOPATH="/go" \
    GOBIN="/usr/local/bin" \
    GO111MODULE="on" \
    CGO_CFLAGS="-g -O2 -Wno-return-local-addr" \
    PROG="flask"

# Copy source to image.
WORKDIR /app
COPY --chown=root:root --chmod="u=rwX,go=rX" . .

# Install system dependencies.
RUN echo 'APT::Acquire::Retries "3";' > /etc/apt/apt.conf.d/80retries && \
    echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/80recommends && \
    echo 'APT::Install-Suggests "false";' > /etc/apt/apt.conf.d/80suggests && \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/80forceyes && \
    echo 'APT::Get::Fix-Missing "true";' > /etc/apt/apt.conf.d/80fixmissing && \
    echo 'force-confold' > /etc/dpkg/dpkg.cfg.d/force-confold && \
    apt-get update && apt-get -qq dist-upgrade && \
    apt-get -qq install \
      curl build-essential libhdf5-dev make bash nano rsync pkg-config util-linux && \
    echo 'alias ll="ls -alh"' >> /etc/skel/.bashrc && \
    echo 'export PS1="\u@$DOCKER_TAG:\w\$ "' >> /etc/skel/.bashrc && \
    chmod u=rwx,go=rx /app/scripts/*.sh && \
    cp /etc/skel/.bashrc /root/.bashrc && \
    cleanup.sh

# Install Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Python dependencies.
RUN requirements.sh

# Expose HTTP service port.
EXPOSE 5000

# Set container entrypoint script.
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Listen at 0.0.0.0:5000 when the container is started.
CMD [ "flask", "--app", "app", "run", "--host", "0.0.0.0"]
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
LABEL org.opencontainers.image.documentation="https://docs.photoprism.app/developer-guide/"
LABEL org.opencontainers.image.authors="PhotoPrism UG <hello@photoprism.app>"
LABEL org.opencontainers.image.vendor="PhotoPrism UG"
LABEL org.opencontainers.image.licenses=""

# Declare build parameters.
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_TAG

# Copy source to image.
WORKDIR /app
COPY --chown=root:root --chmod=755 . .

# Install system dependencies.
RUN echo 'APT::Acquire::Retries "3";' > /etc/apt/apt.conf.d/80retries && \
    echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/80recommends && \
    echo 'APT::Install-Suggests "false";' > /etc/apt/apt.conf.d/80suggests && \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/80forceyes && \
    echo 'APT::Get::Fix-Missing "true";' > /etc/apt/apt.conf.d/80fixmissing && \
    echo 'force-confold' > /etc/dpkg/dpkg.cfg.d/force-confold && \
    apt-get update && apt-get -qq dist-upgrade && \
    apt-get -qq install curl build-essential libhdf5-dev bash nano rsync pkg-config && \
    /app/scripts/cleanup.sh

# Install Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set environment variables.
ENV PATH="/root/.cargo/bin:${PATH}" \
    HDF5_INCLUDEDIR="/usr/include/hdf5"

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose HTTP port.
EXPOSE 5000

# Listen at 0.0.0.0:5000.
CMD [ "flask", "--app", "app", "run", "--host", "0.0.0.0"]
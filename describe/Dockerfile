FROM python:3.12-slim

# Declare build parameters.
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_TAG

# Install build dependencies.
RUN apt-get update && apt-get install -y \
  curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source to image.
WORKDIR /app
COPY . .

# Install Python packages.
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [ "flask", "--app", "app", "run", "--host", "0.0.0.0"]
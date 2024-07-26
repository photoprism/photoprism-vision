FROM python:3.11-bookworm

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements.txt first to use cache
COPY requirements.txt ./

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest 
COPY . .

EXPOSE 5000

CMD [ "flask", "--app", "app", "run"]
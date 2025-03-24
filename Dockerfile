FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY describe/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY describe/ .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

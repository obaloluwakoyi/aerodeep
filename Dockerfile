FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set Python path so milestone1/2/3 are importable as packages
ENV PYTHONPATH=/app

# Non-root user for production
RUN useradd -m -u 1000 aerodeep && chown -R aerodeep:aerodeep /app
USER aerodeep

EXPOSE 8000 8501

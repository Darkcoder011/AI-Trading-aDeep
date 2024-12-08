# Use multi-stage build
FROM node:16 AS frontend-build

# Set working directory
WORKDIR /app

# Copy frontend files
COPY frontend/package*.json ./frontend/
COPY frontend/ ./frontend/

# Build frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Python stage
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    git \
    cmake \
    pkg-config \
    python3-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir cython numpy==1.21.0

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt /app/backend/

# Install Python packages in stages
RUN pip install --no-cache-dir \
    pandas==1.3.0 \
    scipy==1.7.0 \
    scikit-learn==1.0.2

# Install remaining packages
RUN pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir tensorflow==2.12.0

FROM python:3.9-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PATH="/app/backend:${PATH}"
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TORCH_HOME=/app/.torch
ENV TRANSFORMERS_CACHE=/app/.transformers
ENV NUMBA_CACHE_DIR=/app/.numba

# Copy application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build /app/frontend/build

# Create cache directories
RUN mkdir -p /app/.torch /app/.transformers /app/.numba && \
    chmod -R 777 /app/.torch /app/.transformers /app/.numba

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=10 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Command to run the application
CMD cd /app/backend && uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75 --log-level info

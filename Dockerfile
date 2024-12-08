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
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir tensorflow==2.12.0

# Copy the application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build /app/frontend/build

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PATH="/app/backend:${PATH}"
ENV TF_ENABLE_ONEDNN_OPTS=0

# Change to the backend directory
WORKDIR /app/backend

# Expose the port
EXPOSE $PORT

# Health check configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=10 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Command to run the application
CMD ["python", "/app/backend/main.py"]

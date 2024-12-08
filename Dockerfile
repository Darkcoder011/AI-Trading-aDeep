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
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir uvicorn

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Set environment variables
ENV PORT=8000
ENV DEBUG=False
ENV PYTHONUNBUFFERED=1
ENV MAX_WORKERS=1
ENV TIMEOUT=65

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command to run the application
CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 65"]
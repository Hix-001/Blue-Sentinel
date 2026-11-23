# ==============================================================================
# BLUE SENTINEL - Autonomous C4ISR Maritime Intelligence Platform
# Production Container Specification for Python AI/ML & Telemetry Core
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=torch \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for network health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (utilizing PyTorch CPU index to keep image compact & avoid massive CUDA binaries)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained neural weights
COPY backend/ ./backend/
COPY models/ ./models/
COPY telemetry/ ./telemetry/

# Expose standard application port
EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch production ASGI server
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]

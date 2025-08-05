# Stage 1: Builder stage
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  tk \
  tcl \
  curl \
  git \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry==1.8.3 \
  && poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY . .

# Install the application in editable mode
RUN pip install --no-cache-dir -e .

# Stage 2: Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  tk \
  tcl \
  curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /app .

# Copy entrypoint and gunicorn config
COPY docker/entrypoint.sh .
COPY gunicorn.conf.py .

# Set proper permissions
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Expose port for web UI
EXPOSE 5001

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

# =============================================================================
# Dockerfile for Smart Retail Revenue Optimizer ML Pipeline
# Target Environment: Lightweight Batch Inference (Linux / Docker)
# =============================================================================

# Use official lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM regressor on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and mock data generation script
COPY src/ ./src/
COPY generate_mock_data.py .

# Create directories for inputs (mock data) and outputs (models, tables, plots)
# to serve as mount points if needed
RUN mkdir -p mock_data model_output/models model_output/tables model_output/plots

# By default, run the full pipeline (both Demand Forecasting and Promotion Recommendations).
# Users can override this command to run specific scripts or pass arguments.
ENTRYPOINT ["python", "-m", "src.run_pipeline"]

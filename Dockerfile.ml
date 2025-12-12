# Restaurant SaaS ML Training - Docker Image
# For batch model training jobs

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ML
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ML code and data
COPY ml/ ./ml/
COPY data/ ./data/

# Create models directory
RUN mkdir -p ./models

# Run ML training
CMD ["python", "-m", "ml.pipelines.demand_forecasting"]

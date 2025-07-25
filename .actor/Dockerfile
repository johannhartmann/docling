# Build stage for installing dependencies
FROM node:20-slim AS builder

# Install necessary tools and prepare dependencies environment in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /build/bin /build/lib/node_modules \
    && cp /usr/local/bin/node /build/bin/    
# Set working directory
WORKDIR /build

# Create package.json and install Apify CLI in one layer
RUN echo '{"name":"docling-actor-dependencies","version":"1.0.0","description":"Dependencies for Docling Actor","private":true,"type":"module","engines":{"node":">=18"}}' > package.json \
    && npm install apify-cli@latest \
    && cp -r node_modules/* lib/node_modules/ \
    && echo '#!/bin/sh\n/tmp/docling-tools/bin/node /tmp/docling-tools/lib/node_modules/apify-cli/bin/run "$@"' > bin/actor \
    && chmod +x bin/actor \
    # Clean up npm cache to reduce image size
    && npm cache clean --force

# Final stage - build from Python base instead of external image
FROM python:3.11-slim-bookworm

LABEL maintainer="Vaclav Vancura <@vancura>" \
      description="Apify Actor for document processing using Docling" \
      version="1.1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools
    curl \
    wget \
    git \
    procps \
    jq \
    # Libraries for OpenCV and graphics
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # Build tools needed for some Python packages
    build-essential \
    # Additional libraries for bitsandbytes
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    DOCLING_SERVE_HOST=0.0.0.0 \
    DOCLING_SERVE_PORT=5001 \
    DOCLING_SERVE_ARTIFACTS_PATH=/models \
    OMP_NUM_THREADS=4 \
    EASYOCR_MODULE_PATH=/tmp/easyocr-models

# Create necessary directories
RUN mkdir -p /models /tmp/huggingface /tmp/torch && \
    chmod 777 /tmp/huggingface /tmp/torch

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install docling with VLM support
RUN pip install --no-cache-dir docling[vlm]

# Pre-download models (with proper error handling)
ARG MODELS_LIST="layout_docling_v2"
RUN echo "Downloading models..." && \
    HF_HUB_DOWNLOAD_TIMEOUT="90" \
    HF_HUB_ETAG_TIMEOUT="90" \
    docling-tools models download -o "${DOCLING_SERVE_ARTIFACTS_PATH}" ${MODELS_LIST} || \
    echo "Warning: Model download failed, will retry at runtime"

# Create user and set permissions
RUN useradd -m -u 1001 -s /bin/bash docling && \
    chown -R 1001:0 ${DOCLING_SERVE_ARTIFACTS_PATH} && \
    chmod -R g=u ${DOCLING_SERVE_ARTIFACTS_PATH}

# Switch to root temporarily to set up directories
USER root
WORKDIR /app

# Create directories needed by actor
RUN mkdir -p /build-files \
             /tmp \
             /tmp/actor-input \
             /tmp/actor-output \
             /tmp/actor-storage \
             /tmp/apify_input \
             /apify_input \
             /opt/app-root/src/.EasyOCR/user_network \
             /tmp/easyocr-models \
    && chown 1001:1001 /build-files \
    && chown -R 1001:1001 /opt/app-root/src/.EasyOCR \
    && chmod 1777 /tmp \
    && chmod 1777 /tmp/easyocr-models \
    && chmod 777 /tmp/actor-input /tmp/actor-output /tmp/actor-storage /tmp/apify_input /apify_input \
    # Fix for uv_os_get_passwd error in Node.js
    && echo "docling:x:1001:1001:Docling User:/app:/bin/sh" >> /etc/passwd

# Copy only required files
COPY --chown=1001:1001 .actor/actor.sh .actor/actor.sh
COPY --chown=1001:1001 .actor/actor.json .actor/actor.json
COPY --chown=1001:1001 .actor/input_schema.json .actor/input_schema.json
RUN chmod +x .actor/actor.sh

# Copy the build files from builder
COPY --from=builder --chown=1001:1001 /build /build-files

# Switch to non-root user
USER 1001

# Set up TMPFS for temporary files
VOLUME ["/tmp"]

# Create additional volumes for OCR models persistence
VOLUME ["/tmp/easyocr-models"]

# Expose the docling-serve API port
EXPOSE 5001

# Run the actor script
ENTRYPOINT [".actor/actor.sh"]
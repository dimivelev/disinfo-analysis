# Dockerfile

# Stage 1: Build the application and download models
FROM python:3.9-slim as builder

# Set environment variables
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_NO_INTERACTION=1 \
    HF_HOME="/app/.cache/huggingface" \
    SPACY_DATA="/app/.spacy"

WORKDIR /app

# Install system dependencies that might be needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Set the cache directories for the new user
ENV HF_HOME="/home/user/app/.cache/huggingface" \
    SPACY_DATA="/home/user/app/.spacy"
RUN mkdir -p $HF_HOME $SPACY_DATA

# Copy the model download script and run it
COPY --chown=user:user download_models.py .
RUN python download_models.py

# Copy the rest of the application code
COPY --chown=user:user main.py .
COPY --chown=user:user agent.py .


# Stage 2: Create the final, smaller production image
FROM python:3.9-slim

# Set environment variables for the final image
ENV HF_HOME="/home/user/app/.cache/huggingface" \
    SPACY_DATA="/home/user/app/.spacy" \
    PYTHONUNBUFFERED=1

# Create a non-root user and set the working directory
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Copy installed packages and downloaded models from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /home/user/app/.cache /home/user/app/.cache
COPY --from=builder /home/user/app/.spacy /home/user/app/.spacy

# Copy the application code
COPY --chown=user:user main.py .
COPY --chown=user:user agent.py .

# Expose the port the app runs on
ENV PORT=8000

# Command to run the application
CMD ["python", "main.py"]
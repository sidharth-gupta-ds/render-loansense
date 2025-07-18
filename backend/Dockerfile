# Use the official Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for SHAP
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install
COPY backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy backend app code
COPY backend/ .

# Expose port used by FastAPI
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

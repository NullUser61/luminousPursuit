# Use the official CUDA base image from NVIDIA
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    unzip \
    python3-venv \
    libglib2.0-0

# Install any additional dependencies if needed
# For example, if your script requires specific Python packages, you can install them here

# Copy your Python script and any other necessary files into the container
COPY modelCreation.py /app/modelCreation.py

# Set the entry point for the container
#CMD ["python3", "modelCreation.py"]

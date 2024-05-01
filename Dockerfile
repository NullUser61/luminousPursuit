# Use the official CUDA base image from NVIDIA
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages
#RUN apt-get update && \
#    apt-get install -y \
#    git \
#    python3 \
#    python3-pip \
#    wget \
#    unzip \
#    python3-venv \
#    python3-opencv \
#    libglib2.0-0


#RUN mkdir -p ~/miniconda3 && \
#    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#    rm Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to PATH
#ENV PATH="/opt/conda/bin:${PATH}"


# Install any additional dependencies if needed
# For example, if your script requires specific Python packages, you can install them here

# Copy your Python script and any other necessary files into the container
COPY . /app

# Set the entry point for the container
#CMD ["python3", "modelCreation.py"]

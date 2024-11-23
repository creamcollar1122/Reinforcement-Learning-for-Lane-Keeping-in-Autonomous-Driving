FROM python:3.9-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && apt-get clean

# Set the working directory
WORKDIR /workspace

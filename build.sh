#!/bin/bash

# Default architecture is jetson
ARCH=${1:-jetson}

# Validate the architecture argument
if [ "$ARCH" != "jetson" ] && [ "$ARCH" != "amd64" ]; then
    echo "Error: Invalid architecture '$ARCH'. Use 'jetson' or 'amd64'."
    exit 1
fi

# Build the Docker image based on the architecture
echo "Building Docker image for $ARCH..."
if [ "$ARCH" == "jetson" ]; then
    docker build -f Dockerfile.jetson -t roboflow-ppe-web .
elif [ "$ARCH" == "amd64" ]; then
    docker build -f Dockerfile.amd64 -t roboflow-ppe-web .
fi

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Successfully built Docker image 'roboflow-ppe-web' for $ARCH."
else
    echo "Error: Failed to build Docker image for $ARCH."
    exit 1
fi
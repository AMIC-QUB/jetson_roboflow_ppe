# Define build argument for architecture (default: jetson)
ARG ARCH=jetson

# Select base image based on ARCH
FROM nvcr.io/nvidia/l4t-base:r35.1.0 AS jetson-base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS amd64-base

# Use the appropriate base image based on ARCH
FROM ${ARCH}-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CAMERA_TYPE=usb  # Default to USB camera; override with 'realsense' if needed

# Update and install common system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    libusb-1.0-0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libopencv-dev \
    python3-opencv \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install librealsense2 for RealSense camera support (common to both architectures)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && wget -qO- https://raw.githubusercontent.com/IntelRealSense/librealsense/master/scripts/setup_ubuntu.sh | bash \
    && apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    roboflow \
    requests \
    numpy \
    pyrealsense2 \
    flask

# Set working directory
WORKDIR /app

# Copy inference scripts and HTML template
COPY inference_usb.py /app/inference_usb.py
COPY inference_realsense.py /app/inference_realsense.py
COPY templates/index.html /app/templates/index.html

# Expose ports: 9001 for Roboflow inference, 5000 for Flask web server
EXPOSE 9001 5000

# Command to run the appropriate inference script based on CAMERA_TYPE
CMD ["sh", "-c", "if [ \"$CAMERA_TYPE\" = \"realsense\" ]; then python3 inference_realsense.py; else python3 inference_usb.py; fi"]
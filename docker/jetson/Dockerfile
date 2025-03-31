# Use NVIDIA L4T base image for Jetson (adjust version based on your Jetson device)
FROM nvcr.io/nvidia/l4t-base:r35.2.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CAMERA_TYPE=usb

# Install basic dependencies
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
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update library cache
RUN ldconfig

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    roboflow \
    requests \
    numpy \
    flask \
    paho-mqtt \
    psycopg2-binary

# Set working directory and create detections folder
WORKDIR /app
RUN mkdir -p /app/detections && chmod -R 777 /app/detections

# Copy inference scripts and templates/static from project root
COPY ./utils.py /app/utils.py
COPY ./inference_usb.py /app/inference_usb.py
COPY ../templates/ /app/templates/
COPY ../static/ /app/static/

# Expose ports: 5000 for Flask web server (if running web interface on Jetson)
EXPOSE 5000

# Command to run the inference script
CMD ["python3", "inference_usb.py"]
# jetson_roboflow_ppe



Build docker image
```bash
chmod +x build.sh
./build.sh
```

Run the Containers:
Inference container:
```bash
docker run -d --runtime nvidia --network host roboflow/roboflow-inference-server-trt-jetson-5.1.1
```

    For USB Camera (default):
```bash
docker run --runtime nvidia --device /dev/video0:/dev/video0 --network host jetson-roboflow-area_detector-web
```
    For RealSense Camera:

```bash
docker run --runtime nvidia --privileged --network host -e CAMERA_TYPE=realsense jetson-roboflow-area_detector-web
```

Go to 
http://{JETSON_IP}:5000


## Testing on laptop
```bash
chmod +x build.sh
./build.sh
docker run -d --gpus all --network host roboflow/roboflow-inference-server-gpu:latest
```
Inference container:
```bash
docker run -d --gpus all --network host roboflow/roboflow-inference-server-gpu:latest
```

For USB Camera (default):
```bash
docker run -it --gpus all --privileged --device /dev/video4:/dev/video4 --network host \
    -e CAMERA_TYPE=usb \
    -e VIDEO_DEVICE_INDEX=4 \
    roboflow-roi-web    
```
For RealSense Camera:

```bash
docker run --gpus all --privileged --network host -e CAMERA_TYPE=realsense roboflow-area_detector-web
```

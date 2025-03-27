# jetson_roboflow_ppe

Build docker image
```bash
docker build -t jetson-roboflow-realsense-web .
```

Run the Container:

    For USB Camera (default):
```bash
docker run --runtime nvidia --device /dev/video0:/dev/video0 --network host jetson-roboflow-camera-web
```
    For RealSense Camera:

```bash
docker run --runtime nvidia --privileged --network host -e CAMERA_TYPE=realsense jetson-roboflow-camera-web
```

Go to 
http://{JETSON_IP}:5000


## Testing on laptop
```bash
docker build -t roboflow-camera-web --build-arg ARCH=amd64 .
docker run -d --gpus all --network host roboflow/roboflow-inference-server-gpu:latest
```


    For USB Camera (default):
```bash
docker run --gpus all --device /dev/video0:/dev/video0 --network host roboflow-camera-web
```
    For RealSense Camera:

```bash
docker run --gpus all --privileged --network host -e CAMERA_TYPE=realsense roboflow-camera-web
```

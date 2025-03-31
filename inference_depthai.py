import os
import depthai as dai
from utils import app, generate_frames

def get_depthai_frame():
    # Check if DepthAI device is available
    if not dai.Device.getAllAvailableDevices():
        print("No DepthAI devices found. Check USB connection and udev rules.")
        while True:
            yield False, None

    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            yield True, frame  # Always yield exactly two values

# Configure Flask app
app.config['GET_FRAME_FUNC'] = get_depthai_frame

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
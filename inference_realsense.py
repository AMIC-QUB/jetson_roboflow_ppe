import pyrealsense2 as rs
import numpy as np
from utils import app

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

def get_realsense_frame():
    """Get a frame from the RealSense camera."""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return False, None
    frame = np.asanyarray(color_frame.get_data())
    return True, frame

app.config['GET_FRAME_FUNC'] = get_realsense_frame

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        pipeline.stop()
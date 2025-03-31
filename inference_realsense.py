import pyrealsense2 as rs
import numpy as np
from utils import app

def get_frame():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            yield True, frame
    finally:
        pipeline.stop()

# Set the frame function
app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
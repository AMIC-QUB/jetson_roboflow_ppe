import cv2
import numpy as np
import socket
import os
from utils import app

def get_frame():
    # Get the IP camera host and port from environment variables
    ip_camera_host = os.getenv('IP_CAMERA_HOST', '192.168.1.100')
    ip_camera_port = int(os.getenv('IP_CAMERA_PORT', 12345))

    # Create a TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the IP camera's TCP stream
        client_socket.connect((ip_camera_host, ip_camera_port))
        print(f"Connected to TCP stream at {ip_camera_host}:{ip_camera_port}")
    except Exception as e:
        print(f"Error: Could not connect to TCP stream at {ip_camera_host}:{ip_camera_port}. Error: {e}")
        while True:
            yield False, None  # Yield a 2-tuple on failure

    # Buffer to store incoming data
    data = b""
    while True:
        try:
            # Receive data from the socket
            chunk = client_socket.recv(4096)
            if not chunk:
                print("Error: TCP stream closed by the camera.")
                yield False, None
                break

            data += chunk

            # Look for MJPEG frame boundaries (start with 0xFFD8, end with 0xFFD9)
            start = data.find(b'\xff\xd8')
            end = data.find(b'\xff\xd9')

            if start != -1 and end != -1:
                # Extract the JPEG frame
                jpg = data[start:end + 2]
                data = data[end + 2:]  # Remove the processed frame from the buffer

                # Decode the JPEG frame into a NumPy array
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("Error: Failed to decode frame from TCP stream.")
                    yield False, None
                else:
                    yield True, frame

        except Exception as e:
            print(f"Error in TCP stream processing: {e}")
            yield False, None
            break

    # Clean up
    client_socket.close()

# Set the frame function
app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
import cv2
import numpy as np
import socket
import os
from utils import app

def get_frame():
    # Get the TCP server host and port from environment variables
    tcp_server_host = os.getenv('TCP_SERVER_HOST', '0.0.0.0')  # Listen on all interfaces
    tcp_server_port = int(os.getenv('TCP_SERVER_PORT', 12345))

    # Create a TCP server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((tcp_server_host, tcp_server_port))
    server_socket.listen(1)  # Listen for one client connection

    print(f"TCP server listening on {tcp_server_host}:{tcp_server_port}...")

    try:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print(f"Client connected from {client_address}")

        # Buffer to store incoming data
        data = b""
        while True:
            try:
                # Receive data from the client
                chunk = client_socket.recv(4096)
                if not chunk:
                    print("Client disconnected.")
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

    except Exception as e:
        print(f"Error in TCP server: {e}")
        while True:
            yield False, None

    finally:
        client_socket.close()
        server_socket.close()

# Set the frame function
app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
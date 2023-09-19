import cv2
import numpy as np
import tensorflow as tf
import time
import socket
import pickle
import struct
import argparse

# Define a function to send the results back to the client
def send_results(client_socket, results):
    # Serialize the results using pickle
    serialized_results = pickle.dumps(results)
    # Send the length of the serialized results
    client_socket.send(struct.pack('L', len(serialized_results)))
    # Send the serialized results
    client_socket.sendall(serialized_results)

    
# Establish socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.50.96', 8001))
server_socket.listen(1)

# Accept the client connection
connection, client_address = server_socket.accept()
print(f"Connection from {client_address} has been established!")

# Receive data from the client and display it
data = b''
payload_size = struct.calcsize("Q")

start_time = time.time()
frame_count = 0

running_sum = 0
count = 0 

# Build black square to fill in parts of the image that are not transmitted
black_square = np.zeros((120, 160, 3), dtype=np.uint8)

while True:
    while len(data) < payload_size:
        packet = connection.recv(1024)
        if not packet:
            break
        data += packet
    if not data:
        break
    # Unpack the payload size and deserialize the frame
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    

    while len(data) < msg_size:
        data += connection.recv(4*1024)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize the frame and display it
    frame = pickle.loads(frame_data)
   # cv2.imshow("Object Detection", frame)


    # Update FPS counter
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = time.time()
        frame_count = 0
        running_sum += fps
        count += 1

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

#cap.release()
print(running_sum/count)


cv2.destroyAllWindows()

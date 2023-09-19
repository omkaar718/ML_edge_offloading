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

    
parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
  '--cropWidth',
  help='Number of blocks to resize.',
  required=False,
  type=int,
  default=1080)
parser.add_argument(
  '--cropHeight',
  help='Number of blocks to resize.',
  required=False,
  type=int,
  default=1080)
parser.add_argument(
  '--outputDim',
  help='Number of blocks to resize.',
  required=False,
  type=int,
  default=15)


args = parser.parse_args()

# Establish socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.50.96', 8001))
server_socket.listen(1)

# Accept the client connection
connection, client_address = server_socket.accept()
print(f"Connection from {client_address} has been established!")

# Receive data from the client and display it
data = b''
payload_size = struct.calcsize("L")

start_time = time.time()
frame_count = 0

running_sum = 0
count = 0 

# Build black square to fill in parts of the image that are not transmitted
black_square = np.zeros((int(args.cropHeight/args.outputDim), int(args.cropWidth/args.outputDim), 3), dtype=np.uint8)
total_squares = args.outputDim * args.outputDim
print
while True:
#     while len(data) < payload_size:
#         packet = connection.recv(1024)
#   #\      print(packet)
#         if not packet:
#             break
#         data += packet
#     if not data:
#         break
#     # Unpack the payload size and deserialize the frame
#     packed_msg_size = data[:payload_size]
#   #  print(packed_msg_size)
#     data = data[payload_size:]
#     msg_size = struct.unpack("L", packed_msg_size)[0]
    

#     while len(data) < msg_size:
#         data += connection.recv(4*1024)

# #    print(len(data))
#     frame_data = data[:msg_size]
#     data = data[msg_size:]

#     # Deserialize the frame and display it
#     sections = {}

    data_size = connection.recv(4)
    data_size = int.from_bytes(data_size, byteorder='big')

    # Receive the serialized data
    data = b""
    while len(data) < data_size:
        packet = connection.recv(data_size - len(data))
        if not packet:
            break
        data += packet
    sections = {}
    sections_info = pickle.loads(data)
#  print(sections)

    for i in range(total_squares):
        if i in sections_info:
            sections[i] = cv2.resize(sections_info[i], (int(args.cropHeight/args.outputDim),int(args.cropWidth/args.outputDim)))
        else:
            sections[i] = black_square

    height, width, color_channels = (args.cropHeight, args.cropWidth, 3)
    output_image = np.zeros((height, width, color_channels), dtype=np.uint8)

    # Loop over each section in the dictionary
    for section_id, section_image in sections.items():
        # Get the section's dimensions
        section_height, section_width, _ = section_image.shape

        # Calculate the coordinates to place the section in the output image
        row_start = (section_id // args.outputDim) * (height // args.outputDim)
        col_start = (section_id % args.outputDim) * (width // args.outputDim)
        row_end = row_start + section_height
        col_end = col_start + section_width
        
        # Place the section in the output image
        output_image[row_start:row_end, col_start:col_end, :] = section_image

    #print(len(frame))
   # print(frame.shape)
  #  print(frame.shape)
#     # Preprocess the input image
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Display the final image with bounding boxes and class labels
    cv2.imshow("Object Detection", output_image)


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

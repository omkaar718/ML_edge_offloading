import cv2
import socket
import pickle
import struct
import time
import numpy as np
import threading
import json
import argparse
import sys
import psutil
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import tflite_runtime
from tflite_runtime.interpreter import Interpreter
import numpy as np
print(tflite_runtime.__version__)
print(tflite_runtime.interpreter.Interpreter)

# Array Downsizer
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# Center crop function

def center_crop(image, target_height, target_width):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the crop starting positions
    start_x = (original_width - target_width) // 2
    start_y = (original_height - target_height) // 2

    # Calculate the crop ending positions
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Perform the center crop
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image


parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
  '--cropWidth',
  help='crop width.',
  required=False,
  type=int,
  default=1080)
parser.add_argument(
  '--cropHeight',
  help='crop height.',
  required=False,
  type=int,
  default=1080)
parser.add_argument(
  '--outputDim',
  help='output dimension.',
  required=False,
  type=int,
  default=15)
  
parser.add_argument(
  '--testBlocks',
  help='output dimension.',
  required=False,
  type=int,
  default=225)



args = parser.parse_args()

# Initialize bounding boxes and classes array
results = []

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.50.96', 8001))

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture('street.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)


# TFLite interpreter model load

interpreter_quant = Interpreter(model_path="smaller_model_166_0.25.tflite", num_threads=4)
interpreter_quant.allocate_tensors()
input_index = interpreter_quant.get_input_details()[0]["index"]
output_index = interpreter_quant.get_output_details()[0]["index"]


# Define payload size 
data = b''
payload_size = struct.calcsize("Q")
    
# Define variables for the FPS counter
fps_start_time = None
fps_frame_count = 0
fps = 0

#Start Receive thread

running_sum = 0
count = 0

# Define variables for the FPS counter
fps_start_time = None
fps_frame_count = 0
fps = 0


abs_start_time = time.time()
start_time = time.time()
frame_count = 0

running_sum = 0
count = 0 

total_squares = args.outputDim * args.outputDim

send_flag = 0


# Append test file
file1 = ('testresults.csv', "a")
total_model_time = 0
while True:


    # Capture frame-by-frame
    global frame
    ret, frame = cap.read()
    if not ret:
      break
   # print(frame.shape)
    frame = center_crop(frame, args.cropHeight, args.cropWidth)
    model_start_time = time.time()
  # #  if send_flag == 0:
      # Invoke interpreter to get segmented images
    # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # input_image = cv2.resize(rgb_image, (224,224))
    # input_image = np.expand_dims(input_image,axis=0).astype(np.float32)

    # interpreter_quant.set_tensor(input_index, input_image)
    # interpreter_quant.invoke()
    # predictions = interpreter_quant.get_tensor(output_index)

    # out = (np.squeeze(predictions)*255).astype(np.uint8)
    # # #print(out.shape)
# #    out = rebin(out, (2,2))
 # #   cv2.namedWindow("feature map", cv2.WINDOW_NORMAL)
  # #  cv2.resizeWindow("feature map", 448, 448)
   # # cv2.imshow('feature map', out)
    
    # out = out.flatten()
    
    # Split the image into 14 by 14 sections of equal size
    sections = np.array_split(frame, args.outputDim)
    sections = [np.array_split(section, args.outputDim, axis=1) for section in sections]
    sections = np.array(sections).reshape((total_squares, *sections[0][0].shape))
    section_info = {i: sections[i] for i in range(total_squares)}

    sections_send = {}
    for i in range(total_squares):
 #       if out[i] > 128:
      if i < args.testBlocks:
          sections_send[i] = section_info[i]
          
    # Serialize the frame and pack it into a struct
    data = pickle.dumps(sections_send)
    
    data_size = len(data).to_bytes(4, byteorder='big')
    #message = struct.pack("L", len(data)) + data
    client_socket.send(data_size)
    client_socket.send(data)

    # Update FPS counter
    frame_count += 1
    model_finish_time = time.time() - model_start_time
    total_model_time += model_finish_time
  #  print(model_finish_time)
    # elapsed_time = time.time() - start_time
    # if elapsed_time > 1:
        # #print('elapsed time: ' ,elapsed_time)
        # fps = frame_count / elapsed_time
        # print(f"FPS: {fps:.2f}")
        # start_time = time.time()
        # frame_count = 0
        # running_sum += fps
        # count += 1


    #cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or ((time.time() - abs_start_time) > 15):
        break

# Release everything if job is finished

cap.release()
#print(running_sum/count)
print('avg time elapsed: ', total_model_time/frame_count)
cv2.destroyAllWindows()

#client_socket.close()

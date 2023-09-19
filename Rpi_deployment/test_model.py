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
  default=500)
parser.add_argument(
  '--cropHeight',
  help='crop height.',
  required=False,
  type=int,
  default=500)
parser.add_argument(
  '--outputDim',
  help='output dimension.',
  required=False,
  type=int,
  default=16)
  


args = parser.parse_args()

# Initialize bounding boxes and classes array
results = []

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture('street.mp4')
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# TFLite interpreter model load

interpreter_quant = Interpreter(model_path="int_only_face_roi_alpha_240_15.tflite", num_threads=4)
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

start_time = time.time()
frame_count = 0

running_sum = 0
count = 0 

# Video output 
width, height = 448, 448
frame_rate = 10


total_squares = args.outputDim * args.outputDim
total_model_time = 0
send_flag = 0
while True:


    # Capture frame-by-frame
    global frame
    ret, frame = cap.read()
    frame = center_crop(frame, args.cropHeight, args.cropWidth)

    #if send_flag == 0:
      # Invoke interpreter to get segmented images
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(rgb_image, (240,240))
    input_image = np.expand_dims(input_image,axis=0).astype(np.int8)
    model_start_time = time.time()
    interpreter_quant.set_tensor(input_index, input_image)
    interpreter_quant.invoke()
    predictions = interpreter_quant.get_tensor(output_index)
    model_finish_time = time.time() - model_start_time
    total_model_time += model_finish_time

    out = (np.squeeze(predictions)*255).astype(np.uint8)


  # Update FPS counter
    frame_count += 1

  #  cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count > 100:
        break

# Release everything if job is finished

cap.release()
print('avg time elapsed: ', total_model_time/frame_count)
cv2.destroyAllWindows()
client_socket.close()

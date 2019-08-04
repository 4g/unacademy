import cv2
import numpy as np
import socket, time
import sys
import pickle
import struct
import StringIO
import json, imutils

import numpy as np
import cv2
from matplotlib import pyplot as plt


def to_int(im):
  return np.array(im * (255/np.max(im)), dtype=np.uint8)


def remove_green(image):
  img = image.copy()
  h,w = img.shape[0], img.shape[1]
  background = img[0:h//8, 0:w//8]
  green_red_ratio = background[:,:,1]/background[:,:,2]
  min_ratio = np.min(green_red_ratio)
  max_ratio = np.min(green_red_ratio)
  img_ratio = img[:,:,1]/img[:,:,2]
  img_ratio[min_ratio < img_ratio] = 0
  img_ratio = to_int(img_ratio)
  img_ratio[img_ratio > 0] = 1
  for i in range(3):
      img[:,:,i] = image[:,:,i]*img_ratio
  return img


def remove_green1(image):
  img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([36, 25, 25])  # -- Lower range --
  upper = np.array([83, 255, 255])  # -- Upper range --
  mask = cv2.inRange(img, lower, upper)
  mask = ~mask
  res = cv2.bitwise_and(image, image, mask=mask)
  return res


def remove_green_2(imgO):
  img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
  lower = np.array([36, 25, 25])  # -- Lower range --
  upper = np.array([83, 255, 255])  # -- Upper range --
  mask = cv2.inRange(img, lower, upper)

  mask = ~mask
  res = cv2.bitwise_and(imgO, imgO, mask=mask)

  thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
  i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  res = cv2.bitwise_and(imgO, imgO, mask=mask)

  maskN = np.ones(imgO.shape[:2], dtype="uint8") * 255

  for c in contours:
    if 85000 < cv2.contourArea(c) < 10000000:
      cv2.drawContours(maskN, [c], -1, 0, -1)
  res = cv2.bitwise_and(imgO, imgO, mask=~maskN)
  return res


def remove_green_3(imgO):
	kernel = np.ones((3, 3), np.uint8)
	img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
	height, width = img.shape[:2]
	area = height * width
	lower = np.array([36, 25, 25])  # -- Lower range --
	upper = np.array([83, 255, 255])  # -- Upper range --
	mask = cv2.inRange(img, lower, upper)

	mask = ~mask
	res = cv2.bitwise_and(imgO, imgO, mask=mask)

	thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	# ret,thresh = cv2.threshold(thresh, 0.5, 255, 0)
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = imutils.grab_contours(contours)
	res = cv2.bitwise_and(imgO, imgO, mask=mask)

	maskN = np.ones(imgO.shape[:2], dtype="uint8") * 255
	for c in contours:
		# print c
		if area / 10 < cv2.contourArea(c) < area - 100:
			cv2.drawContours(maskN, [c], -1, 0, -1)
	res = cv2.bitwise_and(imgO, imgO, mask=~maskN)
	return res


cap=cv2.VideoCapture("/Users/shubhra/remote-opencv-streaming-live-video/green2.mp4")
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8090))


while(cap.isOpened()):
	ret,frame=cap.read()
	if ret == True:
		memfile = StringIO.StringIO()
		np.save(memfile, frame)
		memfile.seek(0)
		data = json.dumps(memfile.read().decode('latin-1'))
		clientsocket.sendall(struct.pack("L", len(data))+data)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
# clientsocket.close()

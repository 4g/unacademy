import numpy as np
import cv2, os
import socket, time, StringIO, json, struct
import sys, imutils

from PIL import Image

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))


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


def resize_video_3(path):
	cap = cv2.VideoCapture(path)
	cc = 0
	while (cap.isOpened()):
		kernel = np.ones((3, 3), np.uint8)
		try:
			ret, imgO = cap.read()
			img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
			height, width = img.shape[:2]
			area = height * width
		except:
			break
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
		_, jpeg = cv2.imencode('.jpg', res)
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


def resize_video_1(path):
	out_path = SCRIPT_PATH + "/../static/output1.mp4"
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	cap = cv2.VideoCapture(path)
	ret, frame = cap.read()
	h, w, d = frame.shape
	out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(w), int(h)))
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			frame = remove_green(frame)
			out.write(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	# while (cap.isOpened()):
	# 	ret, imgO = cap.read()
	# 	if ret == True:
	# 		img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
	# 		lower = np.array([36, 25, 25])  # -- Lower range --
	# 		upper = np.array([83, 255, 255])  # -- Upper range --
	# 		mask = cv2.inRange(img, lower, upper)
	# 		mask = ~mask
	# 		res = cv2.bitwise_and(imgO, imgO, mask=mask)
	# 		# cv2.imshow('frame', res)
	# 		out.write(res)
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break
	# 	else: break
	# while (cap.isOpened()):
	# 	ret, imgO = cap.read()
	# 	if ret == True:
	# 		img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
	# 		lower = np.array([36, 25, 25])  # -- Lower range --
	# 		upper = np.array([83, 255, 255])  # -- Upper range --
	# 		mask = cv2.inRange(img, lower, upper)
	#
	# 		mask = ~mask
	# 		res = cv2.bitwise_and(imgO, imgO, mask=mask)
	# 		# cv2.imshow('frame',res)
	#
	# 		thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	# 		# ret,thresh = cv2.threshold(thresh, 0.5, 255, 0)
	# 		i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# 		res = cv2.bitwise_and(imgO, imgO, mask=mask)
	#
	# 		maskN = np.ones(imgO.shape[:2], dtype="uint8") * 255
	#
	# 		for c in contours:
	# 			if 85000 < cv2.contourArea(c) < 10000000:
	# 				cv2.drawContours(maskN, [c], -1, 0, -1)
	# 		res = cv2.bitwise_and(imgO, imgO, mask=~maskN)
	#
	# 		# cv2.imshow('frame1', res)
	# 		out.write(res)
	# 		if cv2.waitKey(1) & 0xFF == ord('q'):
	# 			break
	# 	else:
	# 		break

	cap.release()
	out.release()
	cv2.destroyAllWindows()
	return '/static/output1.mp4'


def resize_video(path):
	cap = cv2.VideoCapture(path)
	while (cap.isOpened()):
		ret, imgO = cap.read()
		if ret == True:
			img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
			# (h, w) = img.shape[:2]
			# # calculate the center of the image
			# center = (w / 2, h / 2)
			lower = np.array([36, 25, 25])  # -- Lower range --
			upper = np.array([83, 255, 255])  # -- Upper range --
			mask = cv2.inRange(img, lower, upper)

			mask = ~mask
			res = cv2.bitwise_and(imgO, imgO, mask=mask)
			# cv2.imshow('frame',res)

			thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
			# ret,thresh = cv2.threshold(thresh, 0.5, 255, 0)
			i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			res = cv2.bitwise_and(imgO, imgO, mask=mask)

			maskN = np.ones(imgO.shape[:2], dtype="uint8") * 255

			for c in contours:
				if 85000 < cv2.contourArea(c) < 10000000:
					cv2.drawContours(maskN, [c], -1, 0, -1)
			res = cv2.bitwise_and(imgO, imgO, mask=~maskN)
			_, jpeg = cv2.imencode('.jpg', res)
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: break


def multi_stream_process(path, rotate):
	# cap = cv2.VideoCapture(SCRIPT_PATH + "/../static/green.mp4")
	# cap1 = cv2.VideoCapture(SCRIPT_PATH + "/../static/green.mp4")
	cap = cv2.VideoCapture(path)
	cap1 = cv2.VideoCapture(path)
	while True:
		_, img1 = cap1.read()
		ret, imgO = cap.read()
		if ret == True:
			# img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
			(h, w) = imgO.shape[:2]
			# calculate the center of the image
			center = (w / 2, h / 2)
			#
			# lower = np.array([36, 25, 25])  # -- Lower range --
			# upper = np.array([83, 255, 255])  # -- Upper range --
			# mask = cv2.inRange(img, lower, upper)
			#
			# mask = ~mask
			# res = cv2.bitwise_and(img2, img2, mask=mask)
			# # cv2.imshow('frame',res)
			#
			# thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
			# # ret,thresh = cv2.threshold(thresh, 0.5, 255, 0)
			# i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			# res = cv2.bitwise_and(img2, img2, mask=mask)
			#
			# maskN = np.ones(img2.shape[:2], dtype="uint8") * 255
			#
			# for c in contours:
			# 	if 85000 < cv2.contourArea(c) < 10000000:
			# 		cv2.drawContours(maskN, [c], -1, 0, -1)
			# res = cv2.bitwise_and(img2, img2, mask=~maskN)
			kernel = np.ones((3, 3), np.uint8)
			try:
				img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
				height, width = img.shape[:2]
				area = height * width
			except:
				break
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
				if area / 10 < cv2.contourArea(c) < area - 100:
					cv2.drawContours(maskN, [c], -1, 0, -1)
			res = cv2.bitwise_and(imgO, imgO, mask=~maskN)
			M = cv2.getRotationMatrix2D(center, rotate, 1.0)
			rotated901 = cv2.warpAffine(img1, M, (h, w))
			rotated902 = cv2.warpAffine(res, M, (h, w))

			# scale_percent = 220  # percent of original size
			# width = int(img.shape[1] * scale_percent / 100)
			# height = int(img.shape[0] * scale_percent / 100)
			# dim = (width, height)
			# # resize image
			# resized1 = cv2.resize(rotated901, dim, interpolation=cv2.INTER_AREA)
			# resized2 = cv2.resize(rotated902, dim, interpolation=cv2.INTER_AREA)

			imgf = np.hstack((rotated901, rotated902))
			_, jpeg = cv2.imencode('.jpg', imgf)
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break



def stream_resp(path):
	cap = cv2.VideoCapture(path)
	clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	clientsocket.connect(('localhost', 8090))
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	cap = cv2.VideoCapture(SCRIPT_PATH + "/.." + path)
	ret, frame = cap.read()
	h, w, d = frame.shape
	while (cap.isOpened()):
		ret, imgO = cap.read()
		if ret == True:
			img = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
			lower = np.array([36, 25, 25])  # -- Lower range --
			upper = np.array([83, 255, 255])  # -- Upper range --
			mask = cv2.inRange(img, lower, upper)

			mask = ~mask
			res = cv2.bitwise_and(imgO, imgO, mask=mask)
			# cv2.imshow('frame',res)

			thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
			# ret,thresh = cv2.threshold(thresh, 0.5, 255, 0)
			i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			res = cv2.bitwise_and(imgO, imgO, mask=mask)

			maskN = np.ones(imgO.shape[:2], dtype="uint8") * 255

			for c in contours:
				if 85000 < cv2.contourArea(c) < 10000000:
					cv2.drawContours(maskN, [c], -1, 0, -1)
			res = cv2.bitwise_and(imgO, imgO, mask=~maskN)

			memfile = StringIO.StringIO()
			# print "frame print"
			np.save(memfile, res)
			np.save(memfile, frame)
			memfile.seek(0)
			data = json.dumps(memfile.read().decode('latin-1'))

			clientsocket.sendall(struct.pack("L", len(data)) + data)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	# Release everything if job is finished
	cap.release()
	# cv2.destroyAllWindows()
	# return '/static/output.mp4'


def to_int(im):
	return np.array(im * (255/np.max(im)), dtype=np.uint8)


def remove_green(image):
	img = image.copy()
	h,w = img.shape[0], img.shape[1]
	background = img[0:h//8, 0:w//8]
	green_red_ratio = background[:,:,1]/background[:,:,2]
	min_ratio = np.min(green_red_ratio)
	min_ratio = min_ratio * 1.5
	max_ratio = np.max(green_red_ratio)
	# print (img.shape)
	img[:,:,2] = img[:,:,2] + 1
	with np.errstate(divide='ignore'):
		img_ratio = img[:,:,1]/img[:,:,2]
		img_ratio[min_ratio < img_ratio] = 0
		img_ratio = to_int(img_ratio)
		img_ratio[img_ratio > 0] = 1
		img_ratio = 1 - img_ratio
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		img_ratio = cv2.dilate(img_ratio, kernel, iterations=5)
		img_ratio = cv2.erode(img_ratio, kernel, iterations=5)
		for i in range(3):
			image[:,:,i] = image[:,:,i]*img_ratio
	return image

import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt

img = cv2.imread('D:\Other Key\Desktop\pitch.jpg', 0)
ketqua = img

kernel1 = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
kernel2 = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

img1 = cv2.filter2D(img, -1, kernel1)
img2 = cv2.filter2D(img, -1, kernel2)
edge = img1 + img2
ret, binary = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)

width = binary.shape[1]
height = binary.shape[0]
tmp = int(math.sqrt(width**2 + height**2))

lines = list()

for i in range(500):
	while 1:
		x1 = random.randint(1, height - 1)
		y1 = random.randint(1, width- 1)
		if binary[x1][y1] == 255:
			break
	while 1:
		x2 = random.randint(1, height - 1)
		y2 = random.randint(1, width - 1)
		if binary[x2][y2] == 255:
			break
	count = 0
	for x in range(height):
		for y  in range(width):
			if binary[x][y] == 255:
				if (x - x1)*(y2 - y1) == (y - y1)*(x2 - x1):
					count += 1
	if count > 15:
		lines += [[x1, y1, x2, y2, count]]

cv2.imshow('img', img)
for (x1, y1, x2, y2, count) in lines:
	print(x1, y1, x2, y2, count)
	a = (x1 - x2)
	b = (y1 - y2)
	x3 = x1 + 1000*a
	x4 = x2 - 1000*a
	y3 = y1 + 1000*b
	y4 = y2 - 1000*b
	cv2.line(img,(y3, x3) ,(y4, x4),(0,0,255),2)
		

cv2.imshow('ketqua', ketqua)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('edge', edge)
cv2.imshow('binary', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()


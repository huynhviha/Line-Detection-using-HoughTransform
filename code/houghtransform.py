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
M = np.zeros((tmp, 181), dtype = np.int)

lines = list()

for x in range(height):
	for y in range(width):
		if binary[x][y] == 255:
			for theta in range(181):
				rho = int(x*math.sin(theta*math.pi/180) + y*math.cos(theta*math.pi/180))
				M[rho, theta] += 1


for rho in range(-tmp, tmp):
	for theta in range(181):
		if M[rho, theta] > 100:
			lines += [[rho, theta*math.pi/180]]
		
cv2.imshow('img', img)
for (rho,theta) in lines:
	print(rho, theta)
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('ketqua', ketqua)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('edge', edge)
cv2.imshow('binary', binary)

cv2.imshow('hough', M)
cv2.waitKey(0)
cv2.destroyAllWindows()


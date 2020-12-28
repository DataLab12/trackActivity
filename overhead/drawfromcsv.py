import cv2
import numpy as np 
import csv

def remove_every_other(my_list):
	del my_list[1::2]
	return my_list

points = []
point2 = []
point3 = []

with open("s2bt3_ACT.csv", 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		values = []
		for cell in row:
			if cell.isdigit():
				values.append(int(cell))
			else:
				values.append(0)
		points.append(values)

with open("test.csv", 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		values = []
		for cell in row:
			if cell.isdigit():
				values.append(int(cell))
			else:
				values.append(0)
		point2.append(values)

with open("test2.csv", 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		values = []
		for cell in row:
			if cell.isdigit():
				values.append(int(cell))
			else:
				values.append(0)
		point3.append(values)

img = cv2.imread("ship2b3.png")
height, width, layers = img.shape
scale = 50
size = (int(width * scale / 100), int(height * scale / 100))
vid_out = cv2.VideoWriter('csv_out_new.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 120, size)
center = (0, 0)
radius = 1
thickness = 1

# for row in points:
# 	if row[0] and row[1]:
# 		color = (0, 0, 255)
# 		center = (row[0], row[1])
# 		cv2.circle(img, center, radius, color, thickness)
# 	# if row[2] and row[3]:
# 	# 	color = (150, 0, 0)
# 	# 	center = (row[2], row[3])
# 	# 	cv2.circle(img, center, radius, color, thickness)
# 	frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
# 	vid_out.write(frame)

for row in point2:
	if row[0] and row[1] and row[2]:
		if row[2] == 1: 
			color = (255, 0, 0)
			center = (row[0], row[1])
			cv2.circle(img, center, radius, color, thickness)
		else:
			color = (0, 255, 0)
			center = (row[0], row[1])
			cv2.circle(img, center, radius, color, thickness)
	# if row[2] and row[3]:
	# 	color = (150, 0, 0)
	# 	center = (row[2], row[3])
	# 	cv2.circle(img, center, radius, color, thickness)
	frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
	vid_out.write(frame)

for row in point3:
	row[2] = int((row[1]+row[2])/1.2)
	print(row[2]) 
	if row[0] and row[2] and row[3]:
		if row[3] == 1: 
			color = (0, 255, 255)
			center = (row[0], row[2])
			cv2.circle(img, center, radius, color, thickness)
		else:
			color = (255, 0, 255)
			center = (row[0], row[2])
			cv2.circle(img, center, radius, color, thickness)
	# if row[2] and row[3]:
	# 	color = (150, 0, 0)
	# 	center = (row[2], row[3])
	# 	cv2.circle(img, center, radius, color, thickness)
	frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
	vid_out.write(frame)


output_name = "csvtrajectories_new.png"
cv2.imwrite(output_name, img)



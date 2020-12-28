import cv2
import sys 
import numpy as np
import pyperclip as ppc
from tkinter import filedialog
from tkinter import *

def record_click(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
    	mouseX,mouseY = x,y
    	point = "[" + str(mouseX) + ", " + str(mouseY) + "]"
    	cv2.drawMarker(img, (x, y), (0, 0, 255), markerSize=10, thickness=1)
    	blank = np.zeros((64,172,3), np.uint8)
    	cv2.putText(blank, point, (2, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255))
    	cv2.imshow("Point", blank)
    	k=cv2.waitKey(10) & 0XFF

failed = False
if len(sys.argv) == 2:
	file = str(sys.argv[1])
	img = cv2.imread(file)
	if not img.any():
		failed = True
if len(sys.argv) != 2 or failed:
	root = Tk()
	root.filename = filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("png files","*.png"),("jpeg files","*.jpg"),("all files","*.*")))
	file = root.filename
	img = cv2.imread(file)
	root.destroy()

height, width, layers = img.shape
cv2.namedWindow("Select Points")
cv2.namedWindow("Point")
cv2.moveWindow("Point", width+132, 38)
cv2.setMouseCallback("Select Points",record_click)

points = "["
coord = ""
while(1):
    cv2.imshow("Select Points",img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('\r'):
        break
    elif k == ord('s'):
    	coord = "[" + str(mouseX) + ", " + str(mouseY) + "], "
    	points += coord
    	print(coord[:-2], " -  saved")
    elif k == ord('\b'):
    	points = points[:-len(coord)]
    	print(coord[:-2], " -  removed")

if len(points) > 3:
	points = points[:-2]
points += "]"
print(points)
ppc.copy(points)

import json
import cv2
import csv
import numpy as np

with open ("s2bt3cam10.json") as f:
	data10 = json.load(f)
with open ("s2bt3cam11.json") as f:
	data11 = json.load(f)
with open ("s2bt3cam12.json") as f:
	data12 = json.load(f)
with open ("s2bt3cam14.json") as f:
	data14 = json.load(f)

# Cam10
#src10 = np.array([[0, 480], [854, 480], [0, 11], [427, 11], [854, 11]])
# Cam11
src11 = np.array([[2, 477], [852, 478], [3, 142], [852, 139], [1, 31], [852, 25]])
# Cam12
src12 = np.array([[1, 478], [851, 478], [2, 240], [852, 240], [0, 65], [410, 58], [853, 80]])
# Cam14
src14 = np.array([[0, 478], [851, 479], [853, 173], [0, 173], [0, 35], [426, 13], [852, 32]])

# Cam10 
#dest10 = np.array([[820, 833], [780, 778], [800, 982], [716, 869], [631, 756]])
# Cam11 
dest11 = np.array([[467, 495], [493, 470], [335, 456], [465, 338], [75, 378], [404, 70]])
# Cam12 
dest12 = np.array([[499, 451], [528, 450], [468, 405], [556, 406], [216, 43], [513, 45], [810, 45]])
# Cam14 
dest14 = np.array([[458, 531], [459, 481], [394, 443], [394, 571], [117, 738], [118, 508], [117, 286]])

trajectory1 = {14003, 14002, 11006, 11004, 12016, 12002, 12004, 12006, 12007, 12008, 12011, 12014, 12015}
trajectory2 = {14001, 11003, 12046, 12041, 12037, 12033, 12021, 12025, 12026, 12028, 12031, 12023, 12036, 12042, 12017, 12022}

# generate homography matrices
#h10, status = cv2.findHomography(src10, dest10)
h11, status = cv2.findHomography(src11, dest11)
h12, status = cv2.findHomography(src12, dest12)
h14, status = cv2.findHomography(src14, dest14)

fcsv = open("s2bt3.csv", 'w', newline='')
csvw = csv.writer(fcsv)
csvw.writerow(["B1.x", "B1.y", "B2.x", "B2.y"])

dicts_out = [{}]
by_frame = [{}]
frames = {}
b1 = []
b2 = []
for anno in data12:
	if anno["tracks"]:
		tracks = anno["tracks"]
		frame = anno["frameNo"]
		b1x, b1y, b2x, b2y = 0, 0, 0, 0
		for track in tracks:
			tid = 12000 + track["trackID"]
			centerx = (track["bbox"]["x1"] + track["bbox"]["x2"]) / 2
			centery = (track["bbox"]["y1"] + track["bbox"]["y2"]) / 2
			cp = np.array([[centerx, centery]], dtype='float32')
			cp = np.array([cp])
			cpt = cv2.perspectiveTransform(cp, h12)
			center = (int(cpt[0][0][0]), int(cpt[0][0][1]))
			trans_track = (frame, center)
			if tid not in dicts_out[0]:
				dicts_out[0][tid] = []
			dicts_out[0][tid].append(trans_track)
			if tid in trajectory1:
				traj = 0
				b1x = center[0]
				b1y = center[1]
			elif tid in trajectory2:
				traj = 1
				b2x = center[0]
				b2y = center[1]
			else:
				continue
			if frame not in frames:
				frames[frame] = [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]
			if frames[frame][0][traj] == (-1, -1):
				frames[frame][0][traj] = center
			else:
				updated = [int((x + c) / 2) for x,c in zip(frames[frame][0][traj],center)]
				frames[frame][0][traj] = updated
		if b1x or b1y or b2x or b2y:
			if b1x > 461 and b1y < 366:
				b1.append((b1x, b1y))
			b2.append((b2x, b2y))

offset11 = 0
for anno in data11:
	if anno["tracks"]:
		tracks = anno["tracks"]
		frame = anno["frameNo"]
		frame = str(int(frame) + offset11)
		for track in tracks:
			tid = 11000 + track["trackID"]
			centerx = (track["bbox"]["x1"] + track["bbox"]["x2"]) / 2
			centery = (track["bbox"]["y1"] + track["bbox"]["y2"]) / 2
			cp = np.array([[centerx, centery]], dtype='float32')
			cp = np.array([cp])
			cpt = cv2.perspectiveTransform(cp, h11)
			center = (int(cpt[0][0][0]), int(cpt[0][0][1]))
			trans_track = (frame, center)
			if tid not in dicts_out[0]:
				dicts_out[0][tid] = []
			dicts_out[0][tid].append(trans_track)
			if tid in trajectory1:
				traj = 0
				b1x = center[0]
				b1y = center[1]
			elif tid in trajectory2:
				traj = 1
				b2x = center[0]
				b2y = center[1]
			else:
				continue
			if frame not in frames:
				frames[frame] = [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]
			if frames[frame][1][traj] == (-1, -1):
				frames[frame][1][traj] = center
			else:
				updated = [int((x + c) / 2) for x,c in zip(frames[frame][1][traj],center)]
				frames[frame][1][traj] = updated
		if b1x or b1y or b2x or b2y:
			if b1x > 340 and b1y < 421:
				b1.append((b1x, b1y))
			if b2x < 450 and b2x > 365 and b2y > 393 and b2y < 430:
				b2.append((b2x, b2y))

offset14 = 0
for anno in data14:
	if anno["tracks"]:
		tracks = anno["tracks"]
		frame = anno["frameNo"]
		frame = str(int(frame) + offset14)
		for track in tracks:
			tid = 14000 + track["trackID"]
			centerx = (track["bbox"]["x1"] + track["bbox"]["x2"]) / 2
			centery = (track["bbox"]["y1"] + track["bbox"]["y2"]) / 2
			cp = np.array([[centerx, centery]], dtype='float32')
			cp = np.array([cp])
			cpt = cv2.perspectiveTransform(cp, h14)
			center = (int(cpt[0][0][0]), int(cpt[0][0][1]))
			trans_track = (frame, center)
			if tid not in dicts_out[0]:
				dicts_out[0][tid] = []
			dicts_out[0][tid].append(trans_track)
			if tid in trajectory1:
				traj = 0
				b1x = center[0]
				b1y = center[1]
			elif tid in trajectory2:
				traj = 1
				b2x = center[0]
				b2y = center[1]
			else:
				continue
			if frame not in frames:
				frames[frame] = [[(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)], [(-1, -1), (-1, -1)]]
			if frames[frame][2][traj] == (-1, -1):
				frames[frame][2][traj] = center
			else:
				updated = [int((x + c) / 2) for x,c in zip(frames[frame][2][traj],center)]
				frames[frame][2][traj] = updated
		if b1x or b1y or b2x or b2y:
			if b1x < 360 and b1y > 420:
				b1.append((b1x, b1y))
			b2.append((b2x, b2y))

by_frame[0]["frames"] = frames
with open("s2bt3_frames.json", 'w') as f:
	json.dump(by_frame, f)

with open("scene2btake3.json", 'w') as f:
	json.dump(dicts_out, f)

lb1 = len(b1)
lb2 = len(b2)
pts = max(lb1, lb2)
for i in range(pts):
	if i < lb1 and b1[i] != (0, 0):
		c1 = [b1[i][0]]
		c2 = [b1[i][1]]
	else:
		c1 = ["not visible"]
		c2 = c1
	if i < lb2 and b2[i] != (0, 0):
		c3 = [b2[i][0]]
		c4 = [b2[i][1]]
	else:
		c3 = ["not visible"]
		c4 = c3
	csvw.writerow(c1 + c2 + c3 + c4)
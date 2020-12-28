import pandas as pd 
import cv2
import csv
import numpy as np
import math

# Base paths for CSV files 
path10 = 's2bt3Cam10_AltCord.csv'
path11 = 's2bt3Cam11_AltCord.csv'
path12 = 's2bt3Cam12_AltCord.csv'
path14 = 's2bt3Cam14_AltCord.csv'


# Assignment of homogrophy points 
# Cam10
src10 = np.array([[0, 480], [854, 480], [0, 11], [427, 11], [854, 11]])
# Cam11
src11 = np.array([[2, 477], [852, 478], [3, 142], [852, 139], [1, 31], [852, 25]])
# Cam12
src12 = np.array([[1, 478], [851, 478], [2, 240], [852, 240], [0, 65], [410, 58], [853, 80]])
# Cam14
src14 = np.array([[0, 478], [851, 479], [853, 173], [0, 173], [0, 35], [426, 13], [852, 32]])

# Cam10 
dest10 = np.array([[499, 552], [461, 530], [495, 703], [390, 630], [346, 605]])
# Cam11 
dest11 = np.array([[467, 495], [493, 470], [335, 456], [465, 338], [75, 378], [404, 70]])
# Cam12 
dest12 = np.array([[499, 451], [528, 450], [468, 405], [556, 406], [216, 43], [513, 45], [810, 45]])
# Cam14 
dest14 = np.array([[458, 531], [459, 481], [394, 443], [394, 571], [117, 738], [118, 508], [117, 286]])

# generate homography matrices
h10, status = cv2.findHomography(src10, dest10)
h11, status = cv2.findHomography(src11, dest11)
h12, status = cv2.findHomography(src12, dest12)
h14, status = cv2.findHomography(src14, dest14)

# Array assignment and Counter assignment
CordXZ = []

cam10x = []
cam10z = []
cam11x = []
cam11z = []
cam12x = []
cam12z = []
cam14x = []
cam14z = []
c1 = 0
c2 = 0
c3 = 0
c4 = 0

# Each camera has its own indivudal values and objects

c12 = pd.read_csv(path12 , usecols=[0,1,2,3])
cam12x = abs(c12.XCord)
cam12z = abs(c12.ZCord)
cam12id = abs(c12.ID)
cam12dist = abs(c12.Dist)

for i in cam12x:
    # Homogrophize Basix X/Y/Z Cords
    cp = np.array([[cam12x[c1], cam12z[c1]]])
    cp = np.array([cp])
    cpt = cv2.perspectiveTransform(cp, h12)
    # Homogrophize Distances to CAD
    cp2 = np.array([[cam12x[c1], cam12dist[c1]]])
    cp2 = np.array([cp2])
    cpt2 = cv2.perspectiveTransform(cp2, h12)

    # CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), abs(cpt2[0,0,1]), int(cam12id[c1])))
    CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), int(cam12dist[c1]), int(cam12id[c1])))

    c1+=1

c11 = pd.read_csv(path11 , usecols=[0,1,2,3])
cam11x = abs(c11.XCord)
cam11z = abs(c11.ZCord)
cam11id = abs(c11.ID)
cam11dist = abs(c11.Dist)


for i in cam11x:
    # Homogrophize Basix X/Y/Z Cords
    cp = np.array([[cam11x[c2], cam11z[c2]]])
    cp = np.array([cp])
    cpt = cv2.perspectiveTransform(cp, h11)
    # Homogrophize Distances to CAD
    cp2 = np.array([[cam11x[c2], cam11dist[c2]]])
    cp2 = np.array([cp2])
    cpt2 = cv2.perspectiveTransform(cp2, h11)

    # CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), abs(cpt2[0,0,1]), int(cam11id[c2])))
    CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), int(cam11dist[c2]), int(cam11id[c2])))

    c2+=1


# c10 = pd.read_csv(path10 , usecols=[0,1])
# cam10x = abs(c10.XCord)
# cam10z = abs(c10.ZCord)

# for i in cam10x:
#     cp = np.array([[cam10x[c3], cam10z[c3]]])
#     cp = np.array([cp])
#     cpt = cv2.perspectiveTransform(cp, h10)
#     CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1])))
#     c3+=1


c14 = pd.read_csv(path14 , usecols=[0,1,2,3])
cam14x = abs(c14.XCord)
cam14z = abs(c14.ZCord)
cam14id = abs(c14.ID)
cam14dist = abs(c14.Dist)

for i in cam14x:
    cp = np.array([[cam14x[c4], cam14z[c4]]])
    cp = np.array([cp])
    cpt = cv2.perspectiveTransform(cp, h14)

    cp2 = np.array([[cam14x[c4], cam14dist[c4]]])
    cp2 = np.array([cp2])
    cpt2 = cv2.perspectiveTransform(cp2, h12)

    # CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), abs((cpt2[0,0,1])), int(cam14id[c4])))
    CordXZ.append((int(cpt[0,0,0]), int(cpt[0,0,1]), int(cam14dist[c4]), int(cam14id[c4])))

    c4+=1


np.savetxt('test2.csv', CordXZ , delimiter=',', fmt=['%i','%i','%i','%i'], header='B1.x,B1.z,Dist,ID', comments='')

# merger1 = pd.concat([path12,path11])
# merger2 = pd.concat([path10,path14])

# FinalP = pd.concat([merger1,merger2])

# FinalP.to_csv('s2bt3_altcord.csv', index=False)
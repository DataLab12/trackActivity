import os
import cv2
import numpy as np
import json

#CenterNet
import sys
# insert your path to centerNet-deep-sort
MY_CNPATH = '###/centerNet-deep-sort'
CENTERNET_PATH = MY_CNPATH + '/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = './CenterNet/exp/model_best.pth'
ARCH = 'dla_34'
# Usage: tracking_with_activities.py [video path] [annotation path] [output path/filename]
if (len(sys.argv) == 4):
    myvidpath = sys.argv[1]
    anno_path = sys.argv[2]
    output_path = sys.argv[3]
    #sys.exit()
else: 
    myvidpath = '' #optional: add default video path

vidout = True

#MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
#ARCH = 'resdcn_18'



TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))

#vis_thresh
opt.vis_thresh = 0.5


#input_type
opt.input_type = 'vid'   # for video, 'vid',  for webcam, 'webcam', for ip camera, 'ipcam'

#------------------------------
# for video
opt.vid_path = myvidpath  #
#------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
#------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------


from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


def bbox_to_xywh_cls_conf(bbox):
    person_id = 9
    #confidence = 0.5
    # only person
    bbox = bbox[person_id]

    if any(bbox[:, 4] > opt.vis_thresh):

        bbox = bbox[bbox[:, 4] > opt.vis_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

        return bbox[:, :4], bbox[:, 4]

    else:

        return None, None


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()


        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")


        self.write_video = True

    def open(self, video_path):

        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))

        # video
        else :
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("lastrun.avi", fourcc, 30, (self.im_width, self.im_height))
        #return self.vdo.isOpened()

    def append_activities(self, frames):
        anno_file = open(anno_path)
        annos = json.load(anno_file)
        anno_file.close()
        fps_assumed = 15

        if not annos["activities"]:
            return

        acts = annos["activities"]
        for a in acts:
            act = a["activity_id"]
            start = int(a["START - STOP"][0] * fps_assumed)
            stop = int(a["START - STOP"][1] * fps_assumed)
            for i in range(start, stop):
                new_activity = {}
                new_activity["activityID"] = act
                frames[i]["activities"].append(new_activity)

    def prune_empty_frames(self, frames):
        for f in reversed(frames):
            if not f["activities"] and not f["tracks"]:
                frames.remove(f)

    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        json_frames = []
        while self.vdo.grab():

            frame_no +=1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]
            #im = ori_im[ymin:ymax, xmin:xmax, :]

            #start_center =  time.time()

            results = self.detector.run(im)['results']
            bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results)

            new_frame = {}
            new_frame["frameNo"] = frame_no
            new_frame["activities"] = []
            new_frame["tracks"] = []

            if bbox_xywh is not None:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin)) #xmin, ymin
                    #print (outputs)
                    tracks = []
                    for identity, bb in zip(identities, bbox_xyxy):
                        new_track = {}
                        new_track["trackID"] = int(identity)
                        coords = {}
                        coords["x1"] = int(bb[0])
                        coords["y1"] = int(bb[1])
                        coords["x2"] = int(bb[2])
                        coords["y2"] = int(bb[3])
                        new_track["bbox"] = coords
                        #print (coords)
                        tracks.append(new_track)
                    new_frame["tracks"] = tracks

            json_frames.append(new_frame)        

            end = time.time()
            #print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))

            fps =  1 / (end - start )

            avg_fps += fps
            #print("centernet time: {}s, fps: {}, avg fps : {}".format(end - start, fps,  avg_fps/frame_no))

            if vidout:
            	cv2.imshow("test", ori_im)
            	cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)

        self.append_activities(json_frames)
        self.prune_empty_frames(json_frames)
        with open(output_path, "w") as write_file:
            json.dump(json_frames, write_file)

if __name__ == "__main__":
    import sys
    #new_file =
    # if len(sys.argv) == 1:
    #     print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    # else:
    if vidout:
    	cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    	cv2.resizeWindow("test", 800, 600)

    #opt = opts().init()
    det = Detector(opt)

    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open(myvidpath)
    det.detect()

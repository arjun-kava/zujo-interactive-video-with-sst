from tracker import SSTTracker, TrackerConfig, Track
# from sst_tracker import TrackSet as SSTTracker
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
import argparse
import os
import shutil


def test(model,args):
    track_dir=os.path.join(args.root_dir,"track_dir")
    if os.path.isdir(track_dir):
        shutil.rmtree(track_dir)
    os.mkdir(track_dir)
    image_folder=os.path.join(args.root_dir,"frames")
    detection_file_name=os.path.join(args.root_dir,"detection.txt")
    tracker = SSTTracker(model)
    reader = MOTDataReader(image_folder = image_folder,
            detection_file_name =detection_file_name,
            min_confidence=0.0)
    result = list()
    first_run = True
    print(len(reader))
    keys=reader.detection_group_keys
    print(keys)
    for k, item in enumerate(reader):
        i=keys[k]
        if item is None:
            continue
        
        img = item[0]
        det = item[1]
        if img is None or det is None or len(det)==0:
            continue
        if len(det) > 80:
            det = det[:80, :]
        h, w, _ = img.shape
        original_img=img.copy()
        # if first_run:
        #     vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
        #     first_run = False

        det[:, [2,4]] /= float(w)
        det[:, [3,5]] /= float(h)
        image_org = tracker.update(img, det[:, 2:6], False, i)
        # if not image_org is None:
        #     vw.write(image_org)
        tid=0
        for t in tracker.tracks:
            tid+=1
            n = t.nodes[-1]
            if t.age == 1:
                b = n.get_box(tracker.frame_index-1, tracker.recorder)
                tracking_path=os.path.join(track_dir,str(t.id))
                if not os.path.isdir(tracking_path):
                        os.mkdir(tracking_path) 
                cv2.imwrite(os.path.join(tracking_path,str(i)+"_"+str(tid)+".jpg"),original_img[int(b[1]*h):int(b[1]*h)+ int(b[3]*h),int(b[0]*w):int(b[0]*w) +int(b[2]*w)])
                result.append({
                    "frame_idx":i,
                    "track_id":t.id,
                    "left":b[0]*w,
                    "top":b[1]*h,
                    "width":b[2]*w,
                    "height":b[3]*h
                    })
    return result
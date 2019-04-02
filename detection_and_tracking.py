# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import json
import utils
from test_fashion import test

def run_inference_for_single_image(detection_session,image):
    # Run inference
    output_dict = detection_session["session"].run(detection_session["outputs"]["tensor_dict"],feed_dict={detection_session["placeholders"]["image_tensor"]: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
def run_inference_for_multiple_image(detection_session,args):
    images=[]
    images_paths=[os.path.join(args.query_images_path,path) for path in os.listdir(args.query_images_path)]
    for path in images_paths:
        image= cv2.imread(path)
        output_dict = run_inference_for_single_image(detection_session,image)
        ind=np.argmax(np.array(output_dict['detection_scores']))
        bbox=output_dict['detection_boxes'][ind]
        height, width, channels = image.shape
        box = [bbox[0]*height, bbox[1]*width, bbox[2]*height, bbox[3]*width]
        box=list(map(int,box))
        feature = image[box[0]:box[2],box[1]:box[3]]
        cv2.imwrite(path,feature)
    
def create_detections(frame_idx,predictions,detection_ids,images_embedding,height,width):
    
    detection_list = []
    for i in range(predictions["num_detections"]):
        bbox=predictions['detection_boxes'][i]
        score=predictions['detection_scores'][i]
        box = [bbox[0]*height, bbox[1]*width, bbox[2]*height, bbox[3]*width]
        box=list(map(int,box))
        bbox=[box[1],box[0],box[3]-box[1],box[2]-box[0]]
        index=np.where(detection_ids==i)[0][0]
        feature=images_embedding[index]
        f=open("old1.txt","a")
        f.write("frame_idx : "+str(frame_idx)+" detection_id : "+str(i)+" bbox : "+str(bbox)+"\n")
        detection_list.append(Detection(bbox, score, feature,i))
    return detection_list

def run_inference_for_video(detection_session,args):
    video = cv2.VideoCapture(args.video_path)
    width=int(video.get(3)) 
    height=int(video.get(4))  
    fps=video.get(5) 
    frames=int(video.get(7))  
    aspect_ratio = height / width
    image_shape = (width, height)
    if width>1024:
        image_shape = (1024, int(aspect_ratio * 1024))
    frame_idx=0
    skip_frame=1
    frames_dir_path=os.path.join(args.root_dir,"frames")
    utils.create_dir(frames_dir_path)

    detected_txt_file_path=os.path.join(args.root_dir,"detection.txt")
    det_file=open(detected_txt_file_path,'w')

    if args.take_frames_per_sec>0:
        skip_frame=int(fps/args.take_frames_per_sec)
    last_idx=int(frames)
    temp_images=[]
    temp_idx=0
    temp_fid_list=[]
    temp_original_images=[]
    while frame_idx<last_idx:
        ret, frame = video.read()
        if frame_idx%skip_frame==0:
            image = frame
            if width>1024:
                image=cv2.resize(frame,image_shape)
            temp_images.append(image)
            temp_original_images.append(frame)
            temp_fid_list.append(frame_idx)
            temp_idx+=1
        if temp_idx==10:
            print(np.array(temp_images).shape)
            output_dict = detection_session["session"].run(detection_session["outputs"]["tensor_dict"],feed_dict={detection_session["placeholders"]["image_tensor"]: np.array(temp_images)})
            for i in range(10):  
                detection=False  
                for index,value in enumerate(output_dict['detection_boxes'][i]):
                    if output_dict['detection_scores'][i][index] > args.min_confidence:
                        detection=True
                        box = [value[0]*height, value[1]*width, value[2]*height, value[3]*width]
                        top = int(box[1])
                        left = int(box[0])
                        bottom = int(box[3])
                        right = int(box[2]) 
                        det_file.write(f"{temp_fid_list[i]},-1,{top},{left},{bottom-top},{right-left},{output_dict['detection_scores'][i][index]},-1,-1,-1\n")
                if detection:
                    path=os.path.join(frames_dir_path,str(temp_fid_list[i])+".jpg")
                    cv2.imwrite(path,temp_original_images[i])
            temp_images=[]
            temp_fid_list=[]
            temp_original_images=[]
            temp_idx=0   
            

        frame_idx+=1
    return{
        "video_width":width,
        "video_height":height,
        "video_fps":fps,
        "video_frames":frames
    }
def run(detection_session,tracking_model,args):
    
    run_inference_for_multiple_image(detection_session,args)

    outputs=run_inference_for_video(detection_session,args)
    result=test(tracking_model,args)
    return {
        "all_tracks":result,
        "video_fps":outputs["video_fps"],
        "video_frames":outputs["video_frames"],
        "video_width":outputs["video_width"],
        "video_height":outputs["video_height"]
        }
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--video_path", help="Path for video",
        default=None, required=True)
    parser.add_argument(
        "--query_images_path", help="Path for video",
        default=None, required=True)
    parser.add_argument(
        "--output_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="track/")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.3)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    parser.add_argument(
        "--detection_model_dir", help="Path to frozen detection graph." 
        "This is the actual model that is used for the object detection.",
        default='/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/object_detection(dress)/inference_graph/frozen_inference_graph.pb')
    parser.add_argument(
        "--take_frames_per_sec", help="take_frames_per_sec",
        default=4,type=int)
    parser.add_argument(
        "--detection_labels_path", help="List of the strings that is used to add correct label for each box",
        default='/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/object_detection(dress)/street2shop_label.pbtxt')
    return parser.parse_args()
 # What model to download.

if __name__ == "__main__":
    args = parse_args()
    print(run(args))

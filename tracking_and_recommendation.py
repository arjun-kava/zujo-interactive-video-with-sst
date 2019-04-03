#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
import os
from importlib import import_module
from itertools import count

import h5py
import json
import numpy as np
import tensorflow as tf

from re_ranking_ranklist import re_ranking
from predict import prediction
import json
import detection_and_tracking 
import cv2
import utils
import shutil
def get_image(path):
    image_encoded = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))
    image = tf.expand_dims(image_resized,0)

    return image
def get_image_for_emb(path):
    image_encoded = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))

    return image_resized


def main(detection_session,dist_session,emb_session,tracker_model,args):
    outputs = detection_and_tracking.run(detection_session,tracker_model,args)
    query_image_paths = [os.path.join(args.query_images_path,path) for path in os.listdir(args.query_images_path)]
    query_images = [path for path in os.listdir(args.query_images_path)]
    query_embs=utils.get_embeddings(query_image_paths,emb_session,args)
    match_dict={}
    utils.create_dir(os.path.join(args.root_dir,"final_output"))
    track_dir=os.path.join(args.root_dir,"track_dir")
    for track_id in os.listdir(track_dir):
        gallery_images_path=[os.path.join(track_dir,track_id,path) for path in os.listdir(os.path.join(track_dir,track_id))]
        gallery_embs=utils.get_embeddings(gallery_images_path,emb_session,args)
        recommended_dist,recommended_index=prediction(list(query_embs),np.array(gallery_embs),dist_session,args,rank=1)
        array=np.array(recommended_dist)
        indexes=np.where(array>0)
        flag=True
        temp_index=[]
        max_mean_squerd_error=0
        index=-1
        temp_sum=0
        results=[]
        threshold=19
        if indexes[0].shape[0]==0:
            continue
        for i,j in zip(indexes[0],indexes[1]):
            if flag:
                old_i=i
                temp_index.append([i,j])
                temp_sum+=(threshold-recommended_dist[i][j])
                results.append(f"{i}||{j}||{temp_sum}||{(threshold-recommended_dist[i][j])}\n")
                flag=False
                continue
            if old_i==i:
                temp_index.append([i,j])
                temp_sum+=(threshold-recommended_dist[i][j])
                results.append(f"{i}||{j}||{temp_sum}||{(threshold-recommended_dist[i][j])}\n")
            else:

                mean_squerd_error=temp_sum/len(temp_index)
                if max_mean_squerd_error < mean_squerd_error:
                    max_mean_squerd_error=mean_squerd_error
                    index=old_i
                old_i=i
                temp_index=[[i,j]]
                temp_sum=(threshold-recommended_dist[i][j])
                results.append(f"{i}||{j}||{temp_sum}||{(threshold-recommended_dist[i][j])}||{max_mean_squerd_error}||{mean_squerd_error}\n")
        
        mean_squerd_error=temp_sum/len(temp_index)
        if max_mean_squerd_error < mean_squerd_error:
            max_mean_squerd_error=mean_squerd_error
            index=old_i
        results.append(f"{i}||{j}||{temp_sum}||{(threshold-recommended_dist[i][j])}||{max_mean_squerd_error}||{mean_squerd_error}\n")
        o_file=open(os.path.join(args.root_dir,"final_output",str(track_id)+".txt"),"w")
        o_file.writelines(results)
        if index==-1:
            continue
        cv2.imwrite(os.path.join(args.root_dir,"final_output","query_image"+str(track_id)+".jpg"),cv2.imread(query_image_paths[index]))
        cv2.imwrite(os.path.join(args.root_dir,"final_output","gallery_image"+str(track_id)+".jpg"),cv2.imread(gallery_images_path[0]))
        image_id,_=os.path.splitext(query_images[index])
        match_dict[track_id]=image_id

    return {
        "match_dict":match_dict,
        "all_tracks":outputs["all_tracks"],
        "fps":outputs["video_fps"],
        "frames":outputs["video_frames"],
        "video_width":outputs["video_width"],
        "video_height":outputs["video_height"]
    }



if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate a ReID embedding.')
    args = parser.parse_args()

    with open("args.json", 'r') as f:
        args_resumed = json.load(f)
    for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)
    main(args)
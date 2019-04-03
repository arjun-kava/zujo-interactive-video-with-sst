
from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
import os
import shutil
from argparse import ArgumentParser, FileType
import json
import tracking_and_recommendation
import utils
import time
args = utils.get_args()
detection_session=utils.get_detection_session(args)
dist_session=utils.get_dist_session(args)
emb_session=utils.get_emb_session(args)
tracker_model=utils.load_torch_model(args)


class DetectionApi(Resource):
    def get(self):
        return {"hello":"hello"}
    def post(self):
        json_data = request.get_json(force=True)
        print(json_data)
        start=time.time()
        video_url = json_data['video_url']
        images_data = json_data['images']
        root_dir="root_dir_"+json_data['video_id']
        utils.create_dir(root_dir)
        temp_video_dir = root_dir+'/video_dir'
        utils.create_dir(temp_video_dir)
        temp_images_dir = root_dir+'/images_dir'
        utils.create_dir(temp_images_dir)
        print("video_downloaded.....")
        for image_data in images_data:
            utils.download_image(
                image_data['image_url'], temp_images_dir, image_data['image_id'])
        video_path = utils.download_video(video_url, temp_video_dir)
        print("images_downloaded.....")
        # video_path=video_url
        # temp_images_dir=images_data
        args = utils.get_args(root_dir,video_path, temp_images_dir)
        data = tracking_and_recommendation.main(detection_session,dist_session,emb_session,tracker_model,args)
        final_data = []
        findex = 0
        print(data)
        fps = data["fps"]
        while findex < (len(data["all_tracks"])):
            track = data["all_tracks"][findex]
            frame_id = track['frame_idx']
            
            coordinates = []
            cid = 1
            while True:
                x1 = track['left']
                x2 = track['left']+track['width']
                y1 = track['top']
                y2 = track['top']+track['height']

                
               
                if str(track['track_id']) in list(data['match_dict'].keys()):
                    image_id = data['match_dict'][str(track['track_id'])]
                    coordinates.append({
                        'coordinate_id': cid,
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                        'center_x': (x1+x2)/2,
                        'center_y': (y1+y2)/2,
                        'image_id': image_id,
                        'track_id': track['track_id']
                    })
                    cid += 1
                findex += 1
                if findex >= (len(data["all_tracks"])):
                    break
                track = data["all_tracks"][findex]
                if frame_id != track['frame_idx']:
                    break
            if len(coordinates) > 0:
                for i in range(int(data["fps"]/args.take_frames_per_sec)):
                    final_data.append({
                        "frame_id": frame_id+i,
                        "coordinates": coordinates,
                    })
        video_id = json_data['video_id']
        end=time.time()
        print("time ",(end-start))
        return {"video_id": video_id,"fps": fps,"frames":data["frames"],"video_width":data["video_width"],
        "video_height":data["video_height"],"data": final_data}
        # return {"Abc":"abc"}

if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(DetectionApi, '/')  # Route_1
    app.run(host='0.0.0.0',port='8080')

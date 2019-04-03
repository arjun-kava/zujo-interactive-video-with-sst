from argparse import ArgumentParser, FileType
import json
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
from bs4 import BeautifulSoup
import os
import shutil
import tensorflow as tf
from importlib import import_module
from re_ranking_ranklist import re_ranking
from itertools import count
from layer.sst import build_sst

import torch
import torch.backends.cudnn as cudnn

def load_torch_model(args):
    # load the model
    sst = build_sst('test', 900)
    cudnn.benchmark = True
    sst.load_state_dict(torch.load(args.torch_model))
    sst = sst.cuda()
    sst.eval()
    return sst
def download_video(video_link, video_dir):
    r = requests.get(video_link, stream=True)
    with open(os.path.join(video_dir, video_link.split('/')[-1]), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

    return os.path.join(video_dir, video_link.split('/')[-1])


def download_image(img_link, img_dir, image_id):
    response = requests.get(img_link)
    img = Image.open(BytesIO(response.content))
    # img_array=np.array(img)
    # cv2.imwrite(os.path.join(img_dir,(image_id+'.jpg')),img_array)
    img.save(os.path.join(img_dir, (image_id+'.jpg')))


def get_args(root_dir="root_dir",video_path="demo5.mp4", temp_images_dir="temp_images_dir"):
    parser = ArgumentParser(description='Evaluate a ReID embedding.')
    parser.add_argument("--video_path", default=video_path)
    parser.add_argument("--query_images_path", default=temp_images_dir)
    parser.add_argument("--root_dir", default=root_dir)
    args = parser.parse_args()

    with open("args.json", 'r') as f:
        args_resumed = json.load(f)
    for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

    return args


def create_dir(temp_dir):
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)


def get_detection_session(args):
    # Load frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.detection_model_dir, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        device_count = {'GPU': 1}
        )
    detection_sess= tf.Session(config=config,graph=detection_graph)
    return {
        "session":detection_sess,
        "placeholders":{
            "image_tensor":image_tensor
        },
        "outputs":{
            "tensor_dict":tensor_dict
        }
    }

def get_image_for_emb(path):
    image_encoded = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))

    return image_resized

def get_emb_session(args):
    # Load frozen graph
    emb_graph = tf.Graph()
    with emb_graph.as_default():
        image_paths=tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(get_image_for_emb,num_parallel_calls=args.loading_threads)  
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(1)
        dataset_initilizer = dataset.make_initializable_iterator()
        images = dataset_initilizer.get_next()

        
        print(images.shape)
        model = import_module('nets.' + args.model_name)
        head = import_module('heads.' + args.head_name)
        endpoints, body_prefix = model.endpoints(images, is_training=False)
        with tf.name_scope('head'):
            endpoints = head.head(endpoints, args.embedding_dim, is_training=False)
        saver=tf.train.Saver()
        init_op = tf.initialize_all_variables()
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        device_count = {'GPU': 1}
        )
    print(endpoints['emb'].shape)
    emb_session = tf.Session(config=config,graph=emb_graph) 
    checkpoint = tf.train.latest_checkpoint(args.experiment_root)
    emb_session.run(init_op)
    saver.restore(emb_session, checkpoint)
    print("done")
    return {
        "session":emb_session,
        "placeholders":{
            "image_paths":image_paths
        },
        "dataset_initilizer":dataset_initilizer,
        "outputs":{
            "embedding":endpoints['emb']
        }
    }
def find_index_list(emb,rank):
    index=[]
    dist=[]
    temp_dist = emb.copy()
    for k in range(temp_dist[0].shape[0]):
        temp_index=np.argmin(temp_dist,1)
        index.append(temp_index)
        dist.append(np.min(temp_dist,1))
        for j in range(temp_dist.shape[0]):
            temp_dist[j][temp_index[j]]=np.inf
    return np.array(dist),np.array(index)


def get_image(path):
    image_encoded = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))

    return image_resized    
def get_dist_session(args):
    # Load frozen graph
    dist_graph = tf.Graph()
    with dist_graph.as_default():
        query_embeddings=tf.placeholder(tf.float32, shape=[None,128])
        gallery_embeddings=tf.placeholder(tf.float32,shape=[None,128])
        dataset = tf.data.Dataset.from_tensor_slices(query_embeddings) 
        dataset = dataset.batch(args.dist_batch_size)
        dataset = dataset.prefetch(1)
        dataset_initilizer = dataset.make_initializable_iterator()
        query_embeddings_batch = dataset_initilizer.get_next()
        
        if args.metric=='euclidean':
            diffs=tf.expand_dims(query_embeddings_batch, axis=1) - tf.expand_dims(gallery_embeddings, axis=0)
            dists=tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1))
            recommended_image_dist_list,recommended_image_index_list=tf.py_func(find_index_list,[dists,tf.constant(args.rank)],[tf.float32,tf.int64])

        elif args.metric=='re_ranking':
            recommended_image_dist_list,recommended_image_index_list=tf.py_func(re_ranking, [query_embeddings_batch,gallery_embeddings,tf.constant(args.rank)],[tf.float32,tf.int64])
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        device_count = {'GPU': 1}
        )
    dist_session = tf.Session(config=config,graph=dist_graph) 
    print("done")
    return {
        "session":dist_session,
        "placeholders":{
            "query_embeddings":query_embeddings,
            "gallery_embeddings":gallery_embeddings
        },
        "dataset_initilizer":dataset_initilizer,
        "outputs":{
            "recommended_image_dist_list":recommended_image_dist_list,
            "recommended_image_index_list":recommended_image_index_list
        }
    }

def get_embeddings(image_paths,emb_session,args):
    emb_session["session"].run(emb_session["dataset_initilizer"].initializer,feed_dict={emb_session["placeholders"]["image_paths"]: image_paths})
    all_embeddings=[]
    for start_idx in count(step=args.batch_size):
        try:
            b_shape=args.batch_size
            if start_idx+b_shape>len(image_paths):
                b_shape=len(image_paths)%args.batch_size
            embds = emb_session["session"].run(emb_session["outputs"]["embedding"])
            print('\rPredicted batch {}-{}/{}'.format(start_idx, start_idx + b_shape, len(image_paths)),
                flush=True, end='')
            all_embeddings+=list(embds)
        except tf.errors.OutOfRangeError:
            break  # This just indicates the end of the dataset.
    return np.array(all_embeddings).astype('float32')
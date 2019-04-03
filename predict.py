#!/usr/bin/env python3
from importlib import import_module
from itertools import count
import numpy as np
import tensorflow as tf
from re_ranking_ranklist import re_ranking



def get_image(path):
    image_encoded = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))

    return image_resized
def find_index_list(emb,rank):
    index=[]
    dist=[]
    temp_dist = emb.copy()
    for k in range(rank):
        temp_index=np.argmin(temp_dist,1)
        index.append(temp_index)
        dist.append(np.min(temp_dist,1))
        for j in range(temp_dist.shape[0]):
            temp_dist[j][temp_index[j]]=np.inf
    return np.array(dist),np.array(index)

def prediction(query_embeddings,gallery_embeddings_val,dist_session,args,rank=1,metric='euclidean'):
    dist_session["session"].run(dist_session["dataset_initilizer"].initializer,
    feed_dict={dist_session["placeholders"]["query_embeddings"]: np.array(query_embeddings)})

    # Load the two datasets fully into memory.        
    dist_list=[]
    index_list=[]
    for start_idx in count(step=args.batch_size):
        try:
            b_shape=args.batch_size
            if start_idx+b_shape>len(query_embeddings):
                b_shape=len(query_embeddings)%args.batch_size
            t_index,t_dist = dist_session["session"].run([dist_session["outputs"]["recommended_image_index_list"],dist_session["outputs"]["recommended_image_dist_list"]],
            feed_dict={dist_session["placeholders"]["gallery_embeddings"]:gallery_embeddings_val})
            print('\rPredicted batch {}-{}/{}'.format(
            start_idx, start_idx + b_shape, len(query_embeddings)),
                flush=True, end='')
            t_index=np.reshape(t_index,(gallery_embeddings_val.shape[0],b_shape))
            t_dist=np.reshape(t_dist,(gallery_embeddings_val.shape[0],b_shape))
            t_index=t_index.T
            t_dist=t_dist.T
            dist_list+=list(t_dist)
            index_list+=list(t_index)
        except tf.errors.OutOfRangeError:
            break  # This just indicates the end of the dataset.
    return np.array(dist_list),np.array(index_list)


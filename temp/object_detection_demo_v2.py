import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# What model to download.
MODEL_NAME = '/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/tf_object_detection/inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/tf_object_detection', 'street2shop_label.pbtxt')

# Size, in inches, of the output images.
IMAGE_SIZE = (2, 4)

# Load frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/tf_object_detection/images/dresses/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,f) for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if f.endswith('.jpeg')]

def run_inference_for_single_image(image, graph):

  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      for i in range(0,len(image),32):
        if i ==0:
            output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.array(image[i:i+32])})
            for key in output_dict.keys():
                output_dict[key]=list(output_dict[key])
        else:
            new_output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.array(image[i:i+32])})
            for key in new_output_dict.keys():
                new_output_dict[key]=list(new_output_dict[key])
            for ins,val in new_output_dict.items():
                output_dict[ins]+=val
      # all outputs are float32 numpy arrays, so convert types as appropriate
      # output_dict['num_detections'] = int(output_dict['num_detections'][0])
      # output_dict['detection_classes'] = output_dict[
      #     'detection_classes'][0].astype(np.uint8)
      # output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      # output_dict['detection_scores'] = output_dict['detection_scores'][0]
      # if 'detection_masks' in output_dict:
      #   output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict




cap = cv2.VideoCapture('How To Style A Maxi Dress in 3 Ways.mp4')
i=0
skip_frames=int(cap.get(5)/4)
frames = int(cap.get(7))
aspect_ratio = cap.get(4) / cap.get(3)

image_shape = 1024, int(aspect_ratio * 1024)

original_video = []
while(cap.isOpened()):
    ret, frame = cap.read()
    i+=1
    if i%skip_frames!=0:
      continue    
    print(i)
    if i>frames:
      break
    original_video.append(frame)
cap.release()

output_dict = run_inference_for_single_image(original_video, detection_graph)

for index,img in enumerate(original_video):
    print(index)
    # Actual detection.
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'][index],
        output_dict['detection_classes'][index],
        float(output_dict['detection_scores'][index]),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=15,
        min_score_thresh=0.85)
    cv2.imshow('Object detector', cv2.resize(img,image_shape))
    cv2.resizeWindow('Object detector', image_shape)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
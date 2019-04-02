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


ops = detection_graph.get_operations()
all_tensor_names = {output.name for op in ops for output in op.outputs}
print(all_tensor_names)
# load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/tf_object_detection/images/dresses/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,f) for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if f.endswith('.jpeg')]

def run_inference_for_single_image(image, sess,image_tensor):
  # Run inference
  output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict




i=0
with detection_graph.as_default():
  config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
    device_count = {'GPU': 1}
  )
  with tf.Session(config=config) as sess:
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
    tensor_name="SecondStageBoxPredictor/class_predictions/Conv2D:0"
    tensor_dict['last_layer']=tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    for image_path in TEST_IMAGE_PATHS[:2]:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, sess,image_tensor)
      
      height, width, channels = image_np.shape
      print(image_np.shape)
      print((width,height))
      for index,value in enumerate(output_dict['detection_boxes']):
        # print(output_dict['detection_scores'][index])
        if output_dict['detection_scores'][index] > 0.85:
          print(output_dict['detection_boxes'][index])
          print(output_dict['detection_scores'][index])
          print(output_dict['last_layer'][index].shape)
          box = [value[0]*height, value[1]*width, value[2]*height, value[3]*width]
          top = int(box[1])
          left = int(box[0])
          bottom = int(box[3])
          right = int(box[2])
          cv2.rectangle(image_np,(top,left), (bottom,right), (0, 255, 0), 3)
          cv2.imwrite('as.png',image_np[left:right,top:bottom])
      # vis_util.visualize_boxes_and_labels_on_image_array(
      # image_np,
      # output_dict['detection_boxes'],
      # output_dict['detection_classes'],
      # output_dict['detection_scores'],
      # category_index,
      # instance_masks=output_dict.get('detection_masks'),
      # use_normalized_coordinates=True,
      # line_thickness=15,
      # min_score_thresh=0.85)
      # plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image_np)
      # plt.show()
      cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
      cv2.imshow('Object detector', image_np)
      cv2.resizeWindow('Object detector', 1200,1200)
      cv2.waitKey(0)
cv2.destroyAllWindows()
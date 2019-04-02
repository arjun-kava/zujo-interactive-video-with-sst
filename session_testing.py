import utils
import numpy as np
import tensorflow as tf
numlistarray = lambda x : [float(x) for i in range(20)]
arrays=[np.array(numlistarray(numlist)) for numlist in range(50)]
placeholders = tf.placeholder(tf.float32,shape=[None,20])
dataset = tf.data.Dataset.from_tensor_slices(placeholders)
dataset=dataset.batch(20)
dataset_initilizer = dataset.make_initializable_iterator()
images = dataset_initilizer.get_next()
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(dataset_initilizer.initializer,feed_dict={
        placeholders:np.array(arrays)
    })
    for i in range(3):
        print(sess.run(images).shape)
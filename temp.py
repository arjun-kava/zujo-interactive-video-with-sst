import tensorflow as tf
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]

filenames1 = tf.placeholder(tf.string, shape=[None])

dataset = tf.data.Dataset.from_tensor_slices(filenames1)
# [Other transformations on `dataset`...]
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
# sess.run(iterator.initializer, feed_dict={filenames1: filenames})
# while True:
#     try:
#       print(sess.run(next_element))
#     except tf.errors.OutOfRangeError:
#       break

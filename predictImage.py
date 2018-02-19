
# Author: Samet Kalkan

from PIL import Image
import tools as T
import numpy as np
import os
import tensorflow as tf

MODEL_PATH = "out"


def predict_image(path, sess):
    """
        predicts an image
        Returns:
            path of the image and its class
    """
    img = Image.open(path)
    img = T.resize_and_crop(img, 50)
    img = np.array(img)
    img = img/255.0
    img = img.reshape(1, img.shape[0] * img.shape[1] * img.shape[2])
    output = sess.run("output:0", feed_dict={"input:0": img, "keep_prob:0": 0.5})
    return path, T.classes[np.argmax(output[0])]


def load_frozen_graph():
    filename = MODEL_PATH+"/" + "frozen_mymodel.pb"

    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, [50*50*3], name="input_1")
        drop = tf.placeholder(tf.float32, name="keep_prob_1")

        tf.import_graph_def(
            graph_def,
            # usually, during training you use queues, but at inference time use placeholders
            # this turns into "input
            input_map={"input:0": new_input, "keep_prob:0": drop},
            return_elements=None,
            # if input_map is not None, needs a name
            name="bla",
            op_dict=None,
            producer_op_list=None
        )

    checkpoint_path = tf.train.latest_checkpoint(MODEL_PATH+"/")
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
        saver.restore(sess, checkpoint_path)
        
        #prediction
        y = predict_image("image.jpg" + i, sess)
        print(y)


load_frozen_graph()



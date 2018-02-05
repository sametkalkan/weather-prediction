
# Author: Samet Kalkan

import numpy as np
import tools as T
import tensorflow as tf

validation_data = np.load("../mostnewData/50c/validation_data.npy")
validation_label = np.load("../mostnewData/50c/validation_label.npy")

# normalization
validation_data = validation_data / 255.0

# number of class
num_classes = 4  # Cloudy,Sunny,Rainy,Snowy,Foggy
INPUT_SIZE = 50
CHANNEL_SIZE = 3


def load_frozen_graph():
    filename = "out/" + "frozen_mymodel.pb"
    global validation_data
    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        new_input = tf.placeholder(tf.float32, [INPUT_SIZE*INPUT_SIZE*CHANNEL_SIZE], name="input_1")
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

    checkpoint_path = tf.train.latest_checkpoint("out/")
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
        saver.restore(sess, checkpoint_path)

        evaluate(sess)


def evaluate(sess):
    y = sess.run("output:0", feed_dict={"input:0": validation_data, "keep_prob:0": 1})
    acc = T.get_accuracy_of_class(validation_label, T.get_classes(y))
    print("General Accuracy:", acc)
    for i in range(len(vd)):
        v_data = vd[i][0]
        v_label = vd[i][1]
        y = sess.run("output:0", feed_dict={"input:0": v_data, "keep_prob:0": 1})
        acc = T.get_accuracy_of_class(v_label, T.get_classes(y))
        print(T.classes[i] + ": ", acc)
    print("-----------------------------")


validation_data = validation_data.reshape(validation_data.shape[0],
                                          validation_data.shape[1]*validation_data.shape[2]*validation_data.shape[3])
vd = T.separate_data(validation_data, validation_label)

load_frozen_graph()

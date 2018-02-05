
# Author: Samet Kalkan

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os
import os.path as path
import tools as T
import time

INPUT_SIZE = 50
DEPTH = 3
NUM_CLASSES = 5
MODEL_NAME = "mymodel"
MODEL_PATH = "out"

train_data = np.load("../mostnewData/50c/train_data_n.npy")  # with shape (None, 50, 50, 3)
train_label = np.load("../mostnewData/50c/train_label_n.npy")  # with shape (None,)
train_label_categorical = T.to_categorical(train_label, NUM_CLASSES)  # shape (None, 5)
train_data = train_data/255.0  # normalization

validation_data = np.load("../mostnewData/50c/validation_data.npy")
validation_label = np.load("../mostnewData/50c/validation_label.npy")
validation_label_categorical = T.to_categorical(validation_label, NUM_CLASSES)  # shape (None, 5)
validation_data = validation_data/255.0  # normalization

# reshape it with (None, input_size*input_size*depth), for example (5323, 7500)
train_data = train_data.reshape(train_data.shape[0], INPUT_SIZE*INPUT_SIZE*DEPTH)
validation_data = validation_data.reshape(validation_data.shape[0], INPUT_SIZE*INPUT_SIZE*DEPTH)


def next_batch(batch_size=64, current_offset=0):
    batch_x = train_data[current_offset:current_offset + batch_size]
    batch_y = train_label_categorical[current_offset:current_offset + batch_size]
    return batch_x, batch_y


def get_input():
    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE * INPUT_SIZE * DEPTH], name="input")
    dropout_1 = tf.placeholder(tf.float32, name="keep_prob")
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="expected_output")
    return x, dropout_1, y_


def build_model(x, dropout_1, y_):

    x_image = tf.reshape(x, [-1, INPUT_SIZE, INPUT_SIZE, DEPTH])

    # conv part
    conv1 = tf.layers.conv2d(x_image, 32, 11, 11, 'same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')

    conv2 = tf.layers.conv2d(pool1, 32, 5, 5, 'same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')

    flatten = tf.layers.flatten(pool2)
    # fully connected part
    fc = tf.layers.dense(flatten, 100, activation=tf.nn.relu)
    dropout = tf.nn.dropout(fc, dropout_1)
    logits = tf.layers.dense(dropout, NUM_CLASSES)
    outputs = tf.nn.softmax(logits, name="output")

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    # train step
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    return train_step, outputs


def train(train_step, x, dropout_1, y_, outputs, batch_size=64, epoch=20):
    print("Training is started...")

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, MODEL_PATH, MODEL_NAME + '.pbtxt', True)

        for ep in range(epoch):
            t1 = time.time()
            print("Epoch: %d/%d" % (ep+1, epoch))
            offset = 0
            while offset < len(train_data):
                batch_x, batch_y = next_batch(batch_size, offset)

                sess.run([train_step], feed_dict={x: batch_x, y_: batch_y, dropout_1: 0.5})
                offset += batch_size
            t2 = time.time()
            train_output = sess.run([outputs], feed_dict={x: train_data, y_: train_label_categorical,
                                                          dropout_1: 0.5})
            train_accuracy = T.get_accuracy_of_class(T.get_classes(train_output[0]), train_label)

            val_output = sess.run([outputs], feed_dict={x: validation_data, y_: validation_label_categorical,
                                                        dropout_1: 0.5})
            val_accuracy = T.get_accuracy_of_class(T.get_classes(val_output[0]), validation_label)

            print("  - %.2fs - train acc: %f - val acc: %f" % (t2-t1, train_accuracy, val_accuracy))

        tf.train.Saver().save(sess, MODEL_PATH+"/" + MODEL_NAME + '.chkp')


def save_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph(MODEL_PATH+"/" + MODEL_NAME + '.pbtxt', None, False,
                              MODEL_PATH+"/" + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", MODEL_PATH + "/frozen_" + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(MODEL_PATH+"/frozen_" + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(MODEL_PATH + "/opt_" + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("Graph is saved!")


def main():
    if not path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    x, dropout_1, y_ = get_input()

    train_step, outputs = build_model(x, dropout_1, y_)  # outputs is necessary to calculate accuracy

    train(train_step,  # train placeholder
          x,  # input ph
          dropout_1,  # dropout ph
          y_,  # expected label ph
          outputs,  # predicted output
          batch_size=64,  # batch size
          epoch=40  # epoch
          )

    save_model(["input", "keep_prob"], "output")


main()

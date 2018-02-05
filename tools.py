
# Author: Samet Kalkan

import numpy as np
from PIL import Image
import os
import os.path

classes = ["Cloudy", "Sunny", "Rainy", "Snowy", "Foggy"]


def binary_to_class(label):
    """ Converts a binary class matrix to class vector(integer)
        For example,
        [[0 1 0 0 0]
         [0 0 0 1 0]] => [1 3]
        # Arguments:
            label: matrix to be converted to class vector
    """
    new_lbl = []
    for i in range(len(label)):
        new_lbl.append(np.argmax(label[i]))
    return new_lbl


def to_categorical(label, num_classes):
    new_list = []
    for i in label:
        a = np.zeros(num_classes)
        a[i] = 1
        new_list.append(a)
    return np.array(new_list)


def get_classes(outputs):
    """
        tensorflow yields probabilities such as [0.852431 0.234126 0.432179 0.989094 0.023429] for 5-way classification
        and this function chooses index which has max probability
        for example above class is '3'
    """
    a = []
    for i in range(len(outputs)):
        a.append(np.argmax(outputs[i]))
    return np.array(a)


def get_accuracy_of_class(v_label, y):
    """
        Returns:
            accuracy of given label
        Args:
            v_label: expected outputs
            y: predicted outputs
    """
    c = 0
    for i in range(len(y)):
        if y[i] == v_label[i]:
            c += 1
    return c / len(y)


def separate_data(v_data, v_label):
    """separates validation data and label according to class no
        Args:
            v_data: validation data to be split
            v_label: validation label to be split
        Returns:
            an array that stores '[val_data,val_label]' in each index for each class.
    """
    vd = [[[], []] for _ in range(5)]
    for i in range(len(v_data)):
        cls = int(v_label[i])
        vd[cls][0].append(v_data[i])
        vd[cls][1].append(cls)
    for i in range(5):
        vd[i][0] = np.array(vd[i][0])
        vd[i][1] = np.array(vd[i][1])
    return vd


def resize_and_crop(img, crop_size):
    """
        resizes and crops given image by keeping ratio of image
    Args:
        img: image opened by 'PIL' library
        crop_size: new size of given image
    """
    width = img.size[0]
    height = img.size[1]

    if height > width:
        new_width = crop_size
        new_height = int((crop_size * height)/width)
    else:
        new_width = int((crop_size * width)/height)
        new_height = crop_size

    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    cropped_img = resized_img.crop((0, 0, crop_size, crop_size))
    return cropped_img


def prepare_images(root, dest, size):
    """
        resizes and cropes all images class by class separately
        For example,
        root ( consists of 1024x768 images )
           -- 0 (directory)
              --3234.jpg, 123.jpg, 46345.jpg, 45645.jpg ...
           -- 1 (directory)
              --482.jpg, 925.jpg, 1754.jpg, 2578.jpg ...
           ...
        dest (consists of 50x50 images )
           -- 0 (directory)
              --3234.jpg, 123.jpg, 46345.jpg, 45645.jpg ...
           -- 1 (directory)
              --482.jpg, 925.jpg, 1754.jpg, 2578.jpg ...
           ...

    """
    if not os.path.exists(dest):  # if 'dest' does not exist, makes one.
        os.mkdir(dest)

    for class_name in os.listdir(root):
        if not os.path.exists(dest+class_name):  # if class directory does not exist, makes one.
            os.mkdir(dest+class_name)

        for img_name in os.listdir(root+class_name):  # Images are resized and cropped, then saved
            img = Image.open(root + class_name + "/" + img_name)
            new_img = resize_and_crop(img, size)
            new_img.save(dest + class_name + "/" + img_name)


def image_to_matrix(image_root, dest):
    """
        reads all images in a given directory,
        adds it to an array and labels each image, then saves this model.
    """
    train_data = []
    train_label = []

    for cls in os.listdir(image_root):
        for imageName in os.listdir(image_root + cls):
            img = Image.open(image_root + cls + "/" + imageName)
            img = np.array(img)
            train_data.append(img)
            train_label.append(int(cls))

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data, train_label = shuffle(train_data, train_label)


    np.save(dest+"train_data.npy", train_data)  # model root to save image models(image)
    np.save(dest+"train_label.npy", train_label)  # model root to save image models(label)


def shuffle(data, label):
    r = np.arange(len(data))
    np.random.shuffle(r)
    data = data[r]
    label = label[r]
    return data, label


def split_data(train_data, train_label, ratio):
    x = int(len(train_data) * ratio)
    return train_data[x:], train_label[x:], train_data[:x], train_label[:x]


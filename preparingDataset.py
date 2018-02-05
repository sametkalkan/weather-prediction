# Author: Samet Kalkan

import numpy as np
import tools as T


def split_and_save():
    train_data = np.load("../chosenclasses/model/train_data.npy")
    train_label = np.load("../chosenclasses/model/train_label.npy")

    train_data, train_label = T.shuffle(train_data, train_label)
    train_data, train_label, validation_data, validation_label = T.split_data(train_data, train_label, 0.25)

    np.save("../chosenclasses/model/train_data_n.npy", train_data)
    np.save("../chosenclasses/model/train_label_n.npy", train_label)
    np.save("../chosenclasses/model/validation_data.npy", validation_data)
    np.save("../chosenclasses/model/validation_label.npy", validation_label)


# T.prepare_images("../cektiklerim/deneme/", "../cektiklerim/cropped/", 50)
# T.image_to_matrix("../chosenclasses/images/","../chosenclasses/model/")
# split_and_save()

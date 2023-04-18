from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

from NNData import classes


def set_generator(train, target_dir, image_size, batch_size, i):
    class_labels = list(np.array(os.listdir(target_dir))[i])
    if train:
        t_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    else:
        t_datagen = ImageDataGenerator(rescale=1. / 255)
    t_gen = t_datagen.flow_from_directory(
        target_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=class_labels,
        shuffle=True,
        class_mode='categorical'
    )
    return t_gen


data_dir = os.path.join(
    '101_food_classes_10_percent/'
)
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')
train_set = set_generator(True, train_dir, (224, 224), 32, [0, 1, 2])
test_set = set_generator(False, test_dir, (224, 224), 32, [0, 1, 2])

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


train_set = set_generator(True, '101_food_classes_10_percent/train', (224, 224), batch_size=32, i=classes)
test_set = set_generator(False, '101_food_classes_10_percent/test', (224, 224), batch_size=32, i=classes)

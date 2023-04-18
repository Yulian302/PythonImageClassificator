from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
import random
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 32
epochs = 15

train_generator = train_datagen.flow_from_directory('101_food_classes_10_percent/train',
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory('101_food_classes_10_percent/test',
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='categorical')

history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

model.save('food_classification_model.h5')

model = load_model('food_classification_model.h5')

img_path = '101_food_classes_10_percent/test/lobster_bisque/321266.jpg'
img = Image.open(img_path)
img = img.resize((224, 224))
x = np.expand_dims(img, axis=0)
x = x / 255.0

preds = model.predict(x)
class_idx = np.argmax(preds[0])
class_names = ['apple_pie', 'deviled_eggs', 'lobster_bisque']
class_name = class_names[class_idx]
confidence = preds[0][class_idx]

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_generator = val_datagen.flow_from_directory('101_food_classes_10_percent/test',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 shuffle=False)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

confusion_mtx = confusion_matrix(true_classes, predicted_classes)

cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
cm_display.plot()
plt.show()

# testing
test_dir = '101_food_classes_10_percent/test'
test_classes = os.listdir(test_dir)
random.seed(42)
image_paths = []
for test_class in test_classes:
    class_path = os.path.join(test_dir, test_class)
    class_images = os.listdir(class_path)
    for i in range(5):
        image_paths.append(os.path.join(class_path, random.choice(class_images)))

correct_count = 0
for image_path in image_paths:
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_name = class_names[class_idx]
    confidence = preds[0][class_idx]

    true_class_name = os.path.basename(os.path.dirname(image_path))
    if class_name == true_class_name:
        correct_count += 1

    plt.imshow(img)
    plt.title(f'Prediction: {class_name}, Confidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

accuracy = correct_count / (len(image_paths) * 1.0)
print(f'Accuracy: {accuracy:.2f}')

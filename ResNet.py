from keras.applications.resnet import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the input shape of the images
input_shape = (224, 224, 3)

# Load the pre-trained ResNet50 model without the classification head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Add a new classification head to the model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(3, activation='softmax')(x)

# Freeze the weights of the base model to avoid overfitting
for layer in base_model.layers:
    layer.trainable = False

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Set up the data generators for training and validation
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

# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the trained model
model.save('food_classification_model.h5')

# Load the trained model
model = load_model('food_classification_model.h5')

# Evaluate the model on test data using a confusion matrix
test_generator = val_datagen.flow_from_directory('101_food_classes_10_percent/test',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 shuffle=False)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

confusion_mtx = confusion_matrix(true_classes, predicted_classes)

class_names = list(test_generator.class_indices.keys())
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
cm_display.plot()
plt.show()

# Test the model on a few images
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

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f'Prediction: {class_name}, Confidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

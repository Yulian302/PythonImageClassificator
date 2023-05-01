from keras.applications.resnet import ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


def create_resnet_model(num_layers, num_units, num_classes, fine_tuning=False):
    # Load the pre-trained ResNet50 model without the classification head
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Add a new classification head to the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    for i in range(num_layers):
        x = Dense(num_units, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    # Freeze the weights of the base model to avoid overfitting
    for layer in base_model.layers:
        layer.trainable = False
    if fine_tuning:
        for layer in base_model.layers[:-10]:
            layer.trainable = True
    return model

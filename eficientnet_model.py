from keras.applications.efficientnet import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def create_efficientnet_model(num_classes, fine_tuning=False):
    # Load the pre-trained EfficientNetB0 model without the classification head
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Add a new classification head to the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Freeze the weights of the base model to avoid overfitting
    for layer in base_model.layers:
        layer.trainable = False
    if fine_tuning:
        for layer in base_model.layers[:-10]:
            layer.trainable = True
    predictions = Dense(num_classes, activation='softmax')(x)
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

from keras.optimizers import Adam, SGD

from CNN import create_cnn_model
from NNData import *
from dataSets import *
from tensorboard import tensorboard

cnn_model_adam = create_cnn_model(num_classes, input_shape)
cnn_model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_history_adam = cnn_model_adam.fit(train_set, epochs=epochs, validation_data=test_set,
                                      callbacks=[tensorboard('logs', 'CNN_Adam')])

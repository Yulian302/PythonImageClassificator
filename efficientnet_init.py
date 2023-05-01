from keras.optimizers import Adam, SGD
from NNData import num_classes, input_shape, epochs
from dataSets import *
from eficientnet_model import create_efficientnet_model
from tensorboard_ import tensorboard

efficientnet_model_adam = create_efficientnet_model(num_classes)
efficientnet_model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                                metrics=['accuracy'])
efficientnet_history_adam = efficientnet_model_adam.fit(train_set, epochs=epochs, validation_data=test_set,
                                                        callbacks=[tensorboard('logs', 'Efficientnet_Adam')])

efficientnet_model_sgd = create_efficientnet_model(num_classes)
efficientnet_model_sgd.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_history_sgd = efficientnet_model_sgd.fit(train_set, epochs=epochs, validation_data=test_set,
                                                      callbacks=[tensorboard('logs', 'Efficientnet_SGD')])

efficientnet_model_adam = create_efficientnet_model(num_classes, fine_tuning=True)
efficientnet_model_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                                metrics=['accuracy'])
efficientnet_history_adam = efficientnet_model_adam.fit(train_set, epochs=epochs, validation_data=test_set, callbacks=[
    tensorboard('logs/fine_tuning', 'Efficientnet_Adam')])

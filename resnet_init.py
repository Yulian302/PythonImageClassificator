from keras.optimizers import Adam, SGD

from NNData import *
from ResNet import create_resnet_model
from dataSets import *
from tensorboard_ import tensorboard

resnet_adam = create_resnet_model(num_layers, num_units, num_classes)
resnet_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_history_adam = resnet_adam.fit(train_set, epochs=epochs, validation_data=test_set,
                                      callbacks=[tensorboard('logs', 'Resnet_Adam')])

resnet_sgd = create_resnet_model(num_layers, num_units, num_classes)
resnet_sgd.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_history_sgd = resnet_sgd.fit(train_set, epochs=epochs, validation_data=test_set,
                                    callbacks=[tensorboard('logs', 'Resnet_SGD')])

resnet_adam = create_resnet_model(num_layers, num_units, num_classes)
resnet_adam.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_history_adam = resnet_adam.fit(train_set, epochs=epochs, validation_data=test_set,
                                      callbacks=[tensorboard('logs/fine_tuning', 'Resnet_Adam')])

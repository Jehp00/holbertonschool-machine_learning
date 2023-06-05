#!/usr/bin/env python3
"""Train CNN by classify the CIFAR 10 dataset"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model
    :param X: [numpy.ndarray of shape (m, 32, 32, 3)]:
            contains the CIFAR 10 data where m is the number of data points
    :param Y: [numpy.ndarray of shape (m,)]:
            contains the CIFAR 10 labels for X
    :return: X_p: a numpy.ndarray containing the preprocessed X
        Y_p: a numpy.ndarray containing the preprocessed Y
    """
    x_p = K.applications.densenet.preprocess_input(X,
                                                   data_format="channels_last")
    y_p = K.utils.to_categorical(Y, 10)

    return x_p, y_p


if __name__ == '__name__':
    """
    Trains a convolutional neural network to classify CIFAR 10 dataset
    Saves model to cifar10.h5
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))
    inputs_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          height_factor=(224 // 32),
                                          width_factor=(224 // 32),
                                          data_format="channels_last"))(inputs)

    DenseNet121 = K.applications.DenseNet121(include_top=False,
                                             weights='imagenet',
                                             input_shape=(224, 224, 3))
    activation = K.activations.relu

    X = DenseNet121(inputs_resized, training=False)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(500, activation=activation)(X)
    X = K.layers.Dropout(0.2)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)

    model = K.Model(inputs=inputs, outputs=outputs)

    DenseNet121.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=300,
                        epochs=5, verbose=True)

    model.save('cifar10.h5')

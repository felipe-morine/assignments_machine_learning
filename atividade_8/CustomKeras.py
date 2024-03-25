from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


class CustomKeras:
    """
    Classe modificada do Keras.
    """

    def __init__(self):
        self. model: Sequential = None
        return

    def fit(self, X_train, y_train, nb_epoch=200):
        """
        Treina o modelo
        :param X_train: conjunto de treinamento sem o vetor de classes
        :param y_train: vetor de classes
        :param nb_epoch: numero de epocas
        :return:
        """
        num_features = X_train.shape[1]
        num_classes = y_train.shape[1]

        model = Sequential()

        # camada customizada
        model.add(Dense(32, input_shape=(num_features, )))
        model.add(Activation('tanh'))
        model.add(Dropout(0.25))

        # camada customizada
        model.add(Dense(32))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.25))

        # camada original
        model.add(Dense(150))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Compile
        model.compile(loss='categorical_crossentropy', optimizer='sgd')

        history = model.fit(X_train, y_train, nb_epoch=nb_epoch, verbose=1)
        # print(history)

        self.model = model
        return

    def predict(self, X_test):
        """
        Prediz instâncias desconhecidas
        :param X_test: conjunto de teste
        :return:
        """
        predicts = self.model.predict_classes(X_test)
        return predicts

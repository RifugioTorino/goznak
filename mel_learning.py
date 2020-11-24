import os

import joblib
import keras
import numpy as np
from sklearn.metrics import accuracy_score


class M:
    @staticmethod
    def fit_size_learning(X, new_width):
        '''lex(X)==2'''
        if X.shape[0] != new_width:
            Y = np.copy(X)
            while Y.shape[0] < new_width:
                Y = np.row_stack((Y, X))
            if Y.shape[0] > new_width:
                Y = Y[:new_width, :]
            X = Y
        return X

    def load(self, path='E:\\data\\test\\goznak', tag='train', limit=None, new_width=None):
        X = []
        y = []
        for root, types, _ in os.walk(os.path.join(path, tag)):
            for type in types:
                clean = 1 if type=='clean' else 0
                for root1, clients, _ in os.walk(os.path.join(root, type)):
                    for client in clients[:limit]:
                        for root2, _, files in os.walk(os.path.join(root1, client)):
                            for file in files:
                                data = np.load(os.path.join(root2, file))
                                X.append(data)
                                y.append(clean)

        self.width = new_width or max(x.shape[0] for x in X)
        for i, x in enumerate(X):
            X[i] = self.fit_size_learning(x, self.width)
        X = np.array(X)
        X = X.reshape(*X.shape, 1)
        y = np.array(y)
        return X, y

    def set_model(self, input_shape):
        self.model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                    input_shape=input_shape[1:]),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.Flatten(),  # input_shape=(100,100,3)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(2, activation='softmax')
            ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return self


def script_for_learning(path = 'E:\\data\\test\\goznak',
                        save_model_to_file='noise_classify.pkl',
                        epochs=30,
                        limit=None):
    m = M()
    X_train, y_train = m.load(path=path, tag='train', limit=limit, new_width=None)

    m.set_model(X_train.shape)
    m.model.fit(X_train, y_train, epochs=epochs)
    joblib.dump(m.model, save_model_to_file)

    X_val, y_val = m.load(path=path, tag='val', limit=limit, new_width=m.width)
    acc_val = accuracy_score(m.model.predict_classes(X_val), y_val)
    return acc_val


if __name__ == '__main__':
    try:
        script_for_learning(save_model_to_file='noise_classify.pkl', epochs=1, limit=None)
    except Exception as e:
        print(e)
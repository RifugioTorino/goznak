import os

import numpy as np

from model_unet import unet


class M:

    def __init__(self, path='E:\\data\\test\\goznak', limit=None):
        self.path = path
        self.limit = limit
        self.tag = 'train'
        self.width = 1024
        self.batch = 128

    def gen(self):
        for root, types, _ in os.walk(os.path.join(self.path, self.tag)):
            for root1, clients, _ in os.walk(os.path.join(root, 'clean')):
                for client in clients[:self.limit]:
                    for root_clean, _, files in os.walk(os.path.join(root1, client)):
                        for file in files:
                            clean_path = os.path.join(root_clean, file)
                            noisy_path = clean_path.replace('clean', 'noisy')
                            dirty = np.load(noisy_path)
                            clean = np.load(clean_path)
                            noise = dirty - clean

                            dirty = dirty.reshape(*dirty.shape, 1)
                            noise = noise.reshape(*noise.shape, 1)

                            # to scale between -1 and 1
                            dirty = scaled_in(dirty)
                            noise = scaled_out(noise)
                            for i in range(len(dirty) // self.batch):
                                yield dirty[i*self.batch : (i + 1)*self.batch, :, :],\
                                      noise[i*self.batch : (i + 1)*self.batch, :, :]


def scaled_in(matrix_spec):
    '''global scaling apply to noisy voice spectrograms (scale between -1 and 1)'''
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


def scaled_out(matrix_spec):
    '''global scaling apply to noise models spectrograms (scale between -1 and 1)'''
    matrix_spec = (matrix_spec -6)/82
    return matrix_spec


def inv_scaled_in(matrix_spec):
    '''inverse global scaling apply to noisy voices spectrograms'''
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec


def inv_scaled_out(matrix_spec):
    '''inverse global scaling apply to noise models spectrograms'''
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec


def script_for_learning(path = 'E:\\data\\test\\goznak',
                        save_weights_to_file='den_full.h5',
                        epochs=30,
                        limit=None):
    m = M(path=path, limit=limit)
    nn = unet(input_size=(m.batch, 80, 1))

    X_train, y_train = [], []
    for x, y in m.gen():
        X_train.append(x)
        y_train.append(y)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    nn.fit(X_train, y_train, epochs=epochs, batch_size=32)

    # serialize weights to HDF5
    nn.save_weights(save_weights_to_file)


if __name__ == '__main__':
    try:
        script_for_learning(save_weights_to_file='den_full.h5', epochs=20, limit=None)
    except Exception as e:
        print(e)
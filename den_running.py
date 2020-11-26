import os

import numpy as np
from sklearn.metrics import mean_squared_error

from model_unet import unet
from den_learning import scaled_in, scaled_out, inv_scaled_in, inv_scaled_out


class D:

    def __init__(self, model=None, load_weights_from_file='den_w.h5'):
        self.batch = 128
        if model is None:
            self.model = unet(input_size=(128, 80, 1))
            self.model.load_weights(load_weights_from_file)
        else:
            self.model = model

    def denoising_one(self, dirty_path):
        dirty = np.load(dirty_path)
        clean_path = dirty_path.replace('noisy', 'clean')
        clean = np.load(clean_path)
        # noise = dirty - clean

        dirty_scaled = scaled_in(dirty)
        noise_predicted = None
        for i in range(len(dirty_scaled) // self.batch):
            dirty_piece = dirty_scaled[i * self.batch: (i + 1) * self.batch, :]
            dirty_piece = dirty_piece.reshape(*dirty_piece.shape, 1)
            noise_piece = self.model.predict(np.array([dirty_piece]))
            noise_piece = noise_piece.reshape(*noise_piece.shape[1:3])
            noise_predicted = noise_piece if noise_predicted is None else np.row_stack((noise_predicted, noise_piece))
        clean_predicted = dirty[:len(noise_predicted)] - inv_scaled_out(noise_predicted)
        mse = mean_squared_error(clean[:len(clean_predicted)], clean_predicted)
        return clean[:len(clean_predicted)], clean_predicted, mse

    def denoising_several(self, dirty_folder):
        clean = []
        clean_predicted = []
        mse = []
        for root, clients, _ in os.walk(dirty_folder):
            for client in clients:
                for root1, _, files in os.walk(os.path.join(root, client)):
                    for file in files:
                        try:
                            clean1, clean_predicted1, mse1 = self.denoising_one(os.path.join(root1, file))
                            clean.append(clean1)
                            clean_predicted.append(clean_predicted1)
                            mse.append(mse1)
                        except Exception as e:
                            pass

        return sum(mse)/len(mse)


try:
    load_weights_from_file = 'den_full.h5'
    examples = [
        'E:\\data\\test\\goznak\\train\\noisy\\51\\51_121055_51-121055-0110.npy',
        'E:\\data\\test\\goznak\\train\\noisy\\91\\91_123521_91-123521-0003.npy',
    ]
    d = D(load_weights_from_file=load_weights_from_file)
    for ex in examples:
        clean, clean_predicted, mse = d.denoising_one(ex)

    res = d.denoising_several('E:\\data\\test\\goznak\\val\\noisy')
except Exception as e:
    print(e)


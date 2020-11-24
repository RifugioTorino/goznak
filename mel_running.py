import joblib
import numpy as np


def fit_size_running(X, new_width):
    '''lex(X)==4'''
    if X.shape[1] != new_width:
        Y = np.copy(X)
        while Y.shape[1] < new_width:
            Y = np.concatenate((Y, X), axis=1)
        if Y.shape[1] > new_width:
            Y = Y[:, :new_width, :, :]
        X = Y
    return X


def classify_several(X, model=None, load_model_from_file='noise_classify.pkl'):
    X = X.reshape(*X.shape, 1)
    if model is None:
        model = joblib.load(load_model_from_file)
    return model.predict_classes(fit_size_running(X, model.input_shape[1]))


def classify_one(filename, model=None, load_model_from_file='noise_classify.pkl'):
    data = np.load(filename)
    return classify_several(np.array([data]), model, load_model_from_file)


try:
    classes = {1: 'clean', 0: 'noisy'}
    load_model_from_file = 'noise_classify.pkl'
    examples = [
        'E:\\data\\test\\goznak\\train\\clean\\51\\51_121055_51-121055-0110.npy',
        'E:\\data\\test\\goznak\\train\\clean\\91\\91_123521_91-123521-0003.npy',
        'E:\\data\\test\\goznak\\train\\noisy\\51\\51_121055_51-121055-0110.npy',
        'E:\\data\\test\\goznak\\train\\noisy\\91\\91_123521_91-123521-0003.npy',
    ]
    # Load model for each file
    for ex in examples:
        print(classes[classify_one(ex)[0]])

    # Load model only once
    model = joblib.load(load_model_from_file)
    for ex in examples:
        print(classes[classify_one(ex, model)[0]])
except Exception as e:
    print(e)


# coding: utf-8
from common.np import *
import warnings

def sigmoid(x):
    # print('=== sigmoid BEGIN ===')

    # print(f'P: x.shape: {x.shape}')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            result = 1 / (1 + np.exp(-x))

    except RuntimeWarning:
        print(f'np.max(x): {np.max(x)}')

        x_clipped = np.clip(x, -20, 20)

        # print(f'np.max(x_clipped): {np.max(x_clipped)}')

        result = 1 / (1 + np.exp(-x_clipped))

    # print('=== sigmoid END ===')

    return result

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

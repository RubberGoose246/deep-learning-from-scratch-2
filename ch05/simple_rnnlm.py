# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== SimpleRnnlm.__init__ BEGIN ===')
        print(f'vocab_size: {vocab_size}')
        print(f'wordvec_size: {wordvec_size}')
        print(f'hidden_size: {hidden_size}')

        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        print(f'embed_W.shape: {embed_W.shape}')
        print(f'rnn_Wx.shape: {rnn_Wx.shape}')
        print(f'rnn_Wh.shape: {rnn_Wh.shape}')
        print(f'rnn_b.shape: {rnn_b.shape}')
        print(f'affine_W.shape: {affine_W.shape}')
        print(f'affine_b.shape: {affine_b.shape}')

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        for i, layer in enumerate(self.layers):
            print(f'self.params[{i}].shape: {self.params[i].shape}')
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== SimpleRnnlm.__init__ END ===')

    def forward(self, xs, ts):
        print('=== SimpleRnnlm.forward BEGIN ===')
        print(f'xs.shape: {xs.shape}')
        print(f'ts.shape: {ts.shape}')

        for layer in self.layers:
            xs = layer.forward(xs)

        print(f'xs.shape: {xs.shape}')

        loss = self.loss_layer.forward(xs, ts)

        print('=== SimpleRnnlm.forward END ===')

        return loss

    def backward(self, dout=1):
        print('=== SimpleRnnlm.backward BEGIN ===')

        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        print('=== SimpleRnnlm.backward END ===')

        return dout

    def reset_state(self):
        print('=== SimpleRnnlm.reset_state BEGIN ===')
        self.rnn_layer.reset_state()
        print('=== SimpleRnnlm.reset_state END ===')
